import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from transformers import BertModel


class WordAttention(torch.nn.Module):
    def __init__(
            self,
            # device: str,
            recurrent_size: int,
            attention_dim: int,
            bert_model: str = 'allenai/scibert_scivocab_cased',
    ):
        super().__init__()
        self.attention_dim = attention_dim
        self.recurrent_size = recurrent_size
        # self._device = device
        self.bert_model = BertModel.from_pretrained(bert_model)

        # Maps BERT output to `attention_dim` sized tensor
        self.word_weight = nn.Linear(self.recurrent_size, self.attention_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_weight = nn.Linear(self.attention_dim, 1)

    def recurrent_size(self):
        return self.recurrent_size

    def forward(self, docs, doc_lengths, sent_lengths, attention_masks, token_type_ids, bert_embedding):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :param attention_masks: BERT attention masks; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param token_type_ids: BERT token type IDs; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :return: sentences embeddings, docs permutation indices, docs batch sizes, word attention weights
        """

        # Sort documents by decreasing order in length
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim=0, descending=True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `docs` was of size (num_docs, padded_doc_length, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        docs_valid_bsz = packed_sents.batch_sizes
        print(docs_valid_bsz)
        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_docs, padded_doc_length)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=doc_lengths.tolist(), batch_first=True)

        # Make a long batch of attention masks by removing pad-sentences
        # i.e. `docs` was of size (num_docs, padded_doc_length, padded_sent_length)
        # -> `packed_attention_masks.data` is now of size (num_sents, padded_sent_length)
        packed_attention_masks = pack_padded_sequence(attention_masks, lengths=doc_lengths.tolist(), batch_first=True)

        # Make a long batch of token_type_ids by removing pad-sentences
        # i.e. `docs` was of size (num_docs, padded_doc_length, padded_sent_length)
        # -> `token_type_ids.data` is now of size (num_sents, padded_sent_length)
        packed_token_type_ids = pack_padded_sequence(token_type_ids, lengths=doc_lengths.tolist(), batch_first=True)

        sents, sent_lengths, attn_masks, token_types = (
            packed_sents.data, packed_sent_lengths.data, packed_attention_masks.data, packed_token_type_ids.data
        )

        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]
        print(bert_embedding)
        bert_embedding = self.bert_model(sents, attention_mask=attn_masks, token_type_ids=token_types)

        packed_words = pack_padded_sequence(bert_embedding, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        sentences_valid_bsz = packed_words.batch_sizes

        u_i = torch.tanh(self.word_weight(packed_words.data))
        u_w = self.context_weight(u_i).squeeze(1)
        val = u_w.max()
        att = torch.exp(u_w - val)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, sentences_valid_bsz), batch_first=True)

        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        att_weights = att_weights[sent_unperm_idx]

        return sents, doc_perm_idx, docs_valid_bsz, att_weights


class SentenceAttention(torch.nn.Module):
    def __init__(self, dropout: float, word_recurrent_size: int, recurrent_size: int, attention_dim: int):
        super().__init__()
        # self._device = device
        self.word_recurrent_size = word_recurrent_size
        self.recurrent_size = recurrent_size
        self.dropout = dropout
        self.attention_dim = attention_dim

        assert self.recurrent_size % 2 == 0

        self.encoder = nn.LSTM(
            input_size=self.word_recurrent_size,
            hidden_size=self.recurrent_size // 2,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Maps LSTM output to `attention_dim` sized tensor
        self.sentence_weight = nn.Linear(self.recurrent_size, self.attention_dim)

        # Word context vector (u_w) to take dot-product with
        self.sentence_context_weight = nn.Linear(self.attention_dim, 1)

        self.word_attention = WordAttention(recurrent_size, attention_dim)

    def recurrent_size(self):
        return self.recurrent_size

    def forward(self, docs, doc_lengths, sent_lengths, attention_masks, token_type_ids, bert_embedding):
        """
        :param sent_embeddings: LongTensor (batch_size * padded_doc_length, sentence recurrent dim)
        :param doc_perm_idx: LongTensor (batch_size)
        :param doc_valid_bsz: LongTensor (max_doc_len)
        :param word_att_weights: LongTensor (batch_size * padded_doc_length, max_sent_len)
        :return: docs embeddings, word attention weights, sentence attention weights
        """
        sent_embeddings, doc_perm_idx, doc_valid_bsz, word_att_weights = self.word_attention(docs,
                                                                                             doc_lengths,
                                                                                             sent_lengths,
                                                                                             attention_masks,
                                                                                             token_type_ids,
                                                                                             bert_embedding)

        sent_embeddings = self.dropout(sent_embeddings)

        # Sentence-level LSTM over sentence embeddings
        packed_sentences, _ = self.encoder(PackedSequence(sent_embeddings, doc_valid_bsz))

        u_i = torch.tanh(self.sentence_weight(packed_sentences.data))
        u_w = self.sentence_context_weight(u_i).squeeze(1)
        val = u_w.max()
        att = torch.exp(u_w - val)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, doc_valid_bsz), batch_first=True)

        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as documents by repadding
        docs, _ = pad_packed_sequence(packed_sentences, batch_first=True)

        # Compute document vectors
        docs = docs * sent_att_weights.unsqueeze(2)
        docs = docs.sum(dim=1)

        # Restore as documents by repadding
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, doc_valid_bsz), batch_first=True)

        # Restore the original order of documents (undo the first sorting)
        _, doc_unperm_idx = doc_perm_idx.sort(dim=0, descending=False)
        docs = docs[doc_unperm_idx]

        word_att_weights = word_att_weights[doc_unperm_idx]
        sent_att_weights = sent_att_weights[doc_unperm_idx]

        return docs, word_att_weights, sent_att_weights
