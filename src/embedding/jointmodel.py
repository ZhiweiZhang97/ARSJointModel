import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from transformers import AutoModel
from embedding.loss import FocalLoss, MultiFocalLoss, DiceLoss


def get_index(mask):
    mask = mask.view(mask.size(0), mask.size(1) * mask.size(2))
    index = []
    max_len = 0
    max_len_idx = 0
    for i in range(len(mask)):
        tmp = torch.where(mask[i] != 0)
        index.append(tmp)
        if max_len < len(tmp[0]):
            max_len = len(tmp[0])
            max_len_idx = i
    token_index = []
    max_sentence = index[max_len_idx][0].tolist()[:]
    for i in range(len(index)):
        i_len = len(index[i][0])
        tmp = index[i][0].tolist() + max_sentence[i_len: max_len]
        token_index.append(tmp)
    if mask.size(0) == 1:
        batch_indices = [np.zeros(max_len, dtype=int).tolist()]
    else:
        batch_indices = []
        for i in range(mask.size(0)):
            tmp = np.ones(max_len, dtype=int) * i
            batch_indices += [tmp.tolist()]
    if len(token_index) > 512:
        token_index = token_index[0: 512]
    return torch.tensor(token_index), torch.tensor(batch_indices)


class TimeDistributed(nn.Module):
    def __init__(self, input_size, output_size, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = nn.Linear(input_size, output_size, bias=True)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class ConvLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_channels = hidden_size // 3
        self.max_position_embeddings = 512
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(1, hidden_size),
                               stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(5, hidden_size),
                               stride=1, padding=(2, 0), dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(9, hidden_size),
                               stride=1, padding=(4, 0), dilation=1, groups=1, bias=True)

    def convlution(self, x, conv):
        out = conv(x)  # [batch_size, hidden_dim // 3, sequence_len]
        activation = F.relu(out.squeeze(3))
        out = activation
        return out

    def forward(self, x):
        pooled_output = x.unsqueeze(1)
        h1 = self.convlution(pooled_output, self.conv1)
        h2 = self.convlution(pooled_output, self.conv2)
        h3 = self.convlution(pooled_output, self.conv3)

        pooled_output = torch.cat([h1, h2, h3], 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.permute(0, 2, 1)  # [batch_size, sequence_len, hidden_dim]
        return pooled_output


class LSTM(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(LSTM, self).__init__()
        self.embed_dim = hidden_size
        self.hidden_size = hidden_size // 2
        self.layer_size = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.layer_size, x.size(1), self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.layer_size, x.size(1), self.hidden_size).cuda())
        lstm_output, (_, _) = self.lstm(x, (h_0, c_0))
        return lstm_output


class WordAttention(nn.Module):
    """
    word-level attention. sentence representation for rationale selection.
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super(WordAttention, self).__init__()
        self.word_attention = TimeDistributed(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = TimeDistributed(output_size, 1)

    def forward(self, x, token_mask):
        # valid_abstract = valid_abstract.repeat(token_mask.shape[1], 1).transpose(0, 1).unsqueeze(2)
        att_s = self.dropout(x.view(-1, x.size(-1)))
        att_s = self.word_attention(att_s)
        att_s = self.dropout(torch.tanh(att_s))  # [batch_size * num_sentence * num_token, hidden_dim]  # nan
        raw_att_scores = self.att_scorer(att_s).squeeze(-1).view(x.size(0), x.size(1),
                                                                 x.size(2))  # [batch_size, num_sentence, num_token]
        # raw_att_scores = raw_att_scores.masked_fill(~valid_abstract, -1e4)
        u_w = raw_att_scores.masked_fill((1-token_mask).bool(), float('-inf'))
        # val = u_w.max()
        # att_scores = torch.exp(u_w - val)
        # att_scores = u_w
        # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        att_scores = torch.softmax(u_w, dim=-1)
        # att_scores = torch.mul(sentence_att_score.unsqueeze(2), att_scores)
        # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)  # replace NaN with 0
        # batch_att_scores: word attention scores. [batch_size * num_sentence, num_token]
        batch_att_scores = att_scores.view(-1, att_scores.size(-1))  # word attention weight matrix.
        # print('batch_att_scores: ', batch_att_scores)
        # print('batch_att_scores*att_scores:', torch.mul(sentence_att_score.unsqueeze(2), batch_att_scores))
        # out:  # sentence_representations. [batch_size, num_sentence, hidden_dim]
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
        out = out.view(x.size(0), x.size(1), x.size(-1))  # [batch_size, num_sentence, hidden_dim]
        # print('sentence_reps: ', out)
        # out = torch.mul(sentence_att_score.unsqueeze(2), out)
        # print('sentence_reps*att_scores: ', out)
        # print(100*'-*=')
        mask = token_mask[:, :, 0]  # [batch_size, num_sentence]
        return out, mask, att_scores


class SentenceAttention(nn.Module):
    """
    sentence-level attention. paragraph representation for label prediction.
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super(SentenceAttention, self).__init__()
        self.sentence_attention = TimeDistributed(input_size, output_size)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = TimeDistributed(output_size, 1)
        self.contextualized = False
        self.hidden_size = input_size // 2
        self.lstm = nn.LSTM(input_size, self.hidden_size, 1, dropout=dropout, bidirectional=True)

    def forward(self, sentence_reps, sentence_mask, valid_scores):
        sentence_masks = torch.logical_and(sentence_mask, valid_scores)
        # h_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size).cuda())
        # c_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size).cuda())
        # print(sentence_reps.shape)
        sent_embeddings = self.dropout(sentence_reps)
        # sentence_embedding, (_, _) = self.lstm(sent_embeddings, (h_0, c_0))
        att_s = self.sentence_attention(sent_embeddings)  # [batch_size, num_sentence, hidden_size,]
        u_i = self.dropout(torch.tanh(att_s))  # u_i = tanh(W_s * h_i + b). [batch_size, num_sentence, hidden_size,]
        u_w = self.att_scorer(u_i).squeeze(-1).view(sentence_reps.size(0), sentence_reps.size(1))  # [batch_size, num_sentence]
        # print('u_w: ', u_w)
        att_weights = torch.softmax(u_w.masked_fill((~sentence_mask).bool(), -1e4), dim=-1)
        # print('att_weights: ', att_weights)
        u_w = u_w.masked_fill((~sentence_masks).bool(), -1e4)
        # # att_scores: sentence attention scores. [batch_size, num_sentence]
        att_scores = torch.softmax(u_w, dim=-1)
        # result: abstract representations. [batch_size, hidden_dim]
        result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
        return result, att_weights
        # if sentence_reps.size(0) > 0:
        #     att_scores = F.softmax(sentence_att_scores.masked_fill((~sentence_masks).bool(), -1e4), dim=-1)
        #     sentence_att_scores = torch.softmax(sentence_att_scores, dim=-1)
        #     #att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
        #     result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
        #     return result, sentence_att_scores
        # else:
        #     sentence_att_scores = torch.softmax(sentence_att_scores, dim=-1)
        #     return sentence_reps[:, 0, :], sentence_att_scores


class AbstractAttention(nn.Module):
    """
    word-level attention. abstract representation for abstract retrieval.
    """
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(AbstractAttention, self).__init__()
        self.dense = TimeDistributed(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = ClassificationHead(hidden_size, num_labels, dropout=dropout)
        self.hidden_size = hidden_size
        self.att_scorer = TimeDistributed(hidden_size, 1)
        self.lstm = nn.LSTM(hidden_size, self.hidden_size // 2, 1, dropout=dropout, bidirectional=True)

    def forward(self, x, token_mask, claim_reps):
        att_s = self.dropout(x.view(-1, x.size(-1)))
        att_s = self.dense(att_s)
        att_s = self.dropout(torch.tanh(att_s))  # [batch_size * num_sentence * num_token, hidden_dim]  # nan
        raw_att_scores = self.att_scorer(att_s).squeeze(-1).view(x.size(0), x.size(1),
                                                                 x.size(2))  # [batch_size, num_sentence, num_token]
        u_w = raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf'))
        att_scores = torch.softmax(u_w, dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)  # replace NaN with 0
        # batch_att_scores: word attention scores. [batch_size * num_sentence, num_token]
        word_att_scores = att_scores.view(-1, att_scores.size(-1))  # word attention weight matrix.
        # out:  # sentence_representations. [batch_size, num_sentence, hidden_dim]
        out = torch.bmm(word_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
        sentence_reps = out.view(x.size(0), x.size(1), x.size(-1))  # [batch_size, num_sentence, hidden_dim]
        # sentence_reps = torch.mul(claim_reps.unsqueeze(1), sentence_reps)
        sentence_mask = token_mask[:, :, 0]

        sentence_mask = torch.logical_and(sentence_mask, sentence_mask)
        h_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size // 2).cuda())
        c_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size // 2).cuda())
        # print(sentence_reps.shape)
        sent_embeddings = self.dropout(sentence_reps)
        sentence_embedding, (_, _) = self.lstm(sent_embeddings, (h_0, c_0))
        att_s = self.dense(sentence_embedding)  # [batch_size, num_sentence, hidden_size,]
        u_i = self.dropout(torch.tanh(att_s))  # u_i = tanh(W_s * h_i + b). [batch_size, num_sentence, hidden_size,]
        u_w = self.att_scorer(u_i).squeeze(-1).view(sentence_reps.size(0),
                                                    sentence_reps.size(1))  # [batch_size, num_sentence]
        u_w = u_w.masked_fill((~sentence_mask).bool(), -1e4)
        # sentence_att_scores: sentence attention scores. [batch_size, num_sentence]
        # print('u_w: ', u_w)
        sentence_att_scores = torch.softmax(u_w, dim=-1)
        # result: abstract representations. [batch_size, hidden_dim]
        paragraph_reps = torch.bmm(sentence_att_scores.unsqueeze(1), sentence_reps).squeeze(1)
        claim_paragraph = torch.mul(claim_reps, paragraph_reps)
        output = self.classifier(claim_paragraph)
        sentence_att_scores = sentence_att_scores[:, range(1, sentence_att_scores.shape[1])]
        sentence_att_scores = torch.softmax(sentence_att_scores, dim=-1)
        # print('abstract_sentence_reps: ', sentence_reps)
        return output, sentence_att_scores


class ClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks. """

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = TimeDistributed(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = TimeDistributed(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class SelfAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1):
        super(SelfAttentionNetwork, self).__init__()
        self.dense = TimeDistributed(hidden_dim, hidden_dim)
        self.att_scorer = TimeDistributed(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, token_mask):
        att_s = self.dropout_layer(x)
        att_s = self.dense(att_s)
        u_i = self.dropout_layer(torch.tanh(att_s))
        u_w = self.att_scorer(u_i).squeeze(-1).view(x.size(0), x.size(1))
        u_w = u_w.masked_fill((1 - token_mask).bool(), float('-inf'))
        att_scores = torch.softmax(u_w, dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)
        out = torch.bmm(att_scores.unsqueeze(1), x).squeeze(1)
        return out


def ignore_padding(output, target, padding_idx=2):
    """ Remove padded sentences and label(2). """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = target.view(-1)
    output = output.view(-1, 2)
    idxs = []
    n = len(target)
    for i in range(n):
        if target[i] != padding_idx:
            idxs.append(i)
    new_output = torch.tensor([[0.0, 0.0] for _ in range(len(idxs))])
    new_target = torch.tensor([0 for _ in range(len(idxs))])
    for i, idx in enumerate(idxs):
        new_output[i] = output[idx]
        new_target[i] = target[idx]
    return new_output.to(device), new_target.to(device)


def get_att_label(att_score):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor([[0 if i < 0.25 else 1 for i in score] for score
                         in att_score]).to(device)


class JointModelClassifier(nn.Module):
    def __init__(self, args):
        super(JointModelClassifier, self).__init__()
        self.num_abstract_label = 3
        self.num_rationale_label = 2
        self.sim_label = 2
        self.bert = AutoModel.from_pretrained(args.model)
        self.abstract_criterion = nn.CrossEntropyLoss()
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index=2)
        self.retrieval_criterion = nn.CrossEntropyLoss()
        # self.abstract_criterion = MultiFocalLoss(3, alpha=[0.1, 0.6, 0.3])
        # self.rationale_criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
        # self.retrieval_criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
        self.retrieval_rationale_criterion = nn.BCELoss()
        self.dropout = args.dropout
        self.hidden_dim = args.hidden_dim
        self.word_attention = WordAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.sentence_attention = SentenceAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.rationale_linear = ClassificationHead(self.hidden_dim, self.num_rationale_label, dropout=self.dropout)
        self.abstract_linear = ClassificationHead(self.hidden_dim, self.num_abstract_label, dropout=self.dropout)
        self.abstract_retrieval = AbstractAttention(self.hidden_dim, self.sim_label, dropout=self.dropout)
        self.self_attention = SelfAttentionNetwork(self.hidden_dim, dropout=self.dropout)

        self.extra_modules = [
            self.word_attention,
            self.sentence_attention,
            self.abstract_linear,
            self.rationale_linear,
            self.abstract_criterion,
            self.rationale_criterion,
            self.retrieval_criterion,
        ]

    def reinitialize(self):
        self.word_attention = WordAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.sentence_attention = SentenceAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.rationale_linear = ClassificationHead(self.hidden_dim, self.num_rationale_label, dropout=self.dropout)
        self.abstract_linear = ClassificationHead(self.hidden_dim, self.num_abstract_label, dropout=self.dropout)
        self.abstract_retrieval = AbstractAttention(self.hidden_dim, self.sim_label, dropout=self.dropout)
        self.self_attention = SelfAttentionNetwork(self.hidden_dim, dropout=self.dropout)
        self.extra_modules = [
            self.word_attention,
            self.sentence_attention,
            self.abstract_linear,
            self.rationale_linear,
            self.abstract_criterion,
            self.rationale_criterion,
            self.retrieval_criterion,
        ]

    def forward(self, encoded_dict, transformation_indices, abstract_label=None, rationale_label=None,
                retrieval_label=None, train=False, retrieval_only=False, rationale_sample=1):
        batch_indices, indices_by_batch, mask = transformation_indices
        # match_batch_indices, match_indices_by_batch, match_mask = match_indices
        # (batch_size, num_sep, num_token)
        # print(encoded_dict['input_ids'].shape, batch_indices.shape, indices_by_batch.shape, mask.shape)
        bert_out = self.bert(**encoded_dict)[0]  # [batch_size, sequence_len, hidden_dim]

        title_abstract_token = range(1, batch_indices.shape[1])
        title_abstract_tokens = bert_out[batch_indices[:, title_abstract_token, :],
                                         indices_by_batch[:, title_abstract_token, :], :]
        title_abstract_mask = mask[:, title_abstract_token, :]

        claim_token = bert_out[batch_indices[:, 0, :], indices_by_batch[:, 0, :], :]
        claim_mask = mask[:, 0, :]
        claim_representation = self.self_attention(claim_token, claim_mask)

        sentence_token = range(2, batch_indices.shape[1])
        batch_indices, indices_by_batch, mask = batch_indices[:, sentence_token, :], \
                                                indices_by_batch[:, sentence_token, :], mask[:, sentence_token, :]

        # corpus_tokens = bert_out[match_batch_indices, match_indices_by_batch, :]
        # claim_token = bert_out[match_batch_indices[:, 0, :], match_indices_by_batch[:, 0, :], :]
        # corpus_tokens = corpus_tokens.squeeze(1)  # [batch_size, sequence_len, hidden_dim]
        # claim_token = claim_token.squeeze(1)
        # # cnn_out = self.conv_layer(corpus_tokens)
        # # classifier_out = output[1]  # [batch_size, hidden_dim]
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]  # get Bert tokens(sentences level)
        # bert_tokens: [batch_size, num_sentence, num_token, hidden_dim]
        # abstract_retrieval = self.abstract_retrieval(corpus_tokens, cnn_out)
        abstract_retrieval, sentence_att_scores = self.abstract_retrieval(title_abstract_tokens,
                                                                          title_abstract_mask, claim_representation)
        # print('abstract: ', abstract_label, 'retrieval: ', retrieval_label)
        # print('rationale_label: ', rationale_label)
        # print('abstract_sentence_att_scores: ', sentence_att_scores)
        # if abstract_retrieval[:, 1] > abstract_retrieval[:, 0]:
        #     print('True')
        # else:
        #     print('False')

        retrieval_out = torch.argmax(abstract_retrieval.cpu(), dim=-1).detach().numpy().tolist()
        if retrieval_only:
            retrieval_loss = self.retrieval_criterion(abstract_retrieval,
                                                      retrieval_label) if retrieval_label is not None else None
            return retrieval_out, retrieval_loss

        sentence_representations, sentence_mask, word_att_score = self.word_attention(bert_tokens, mask)
        # print('word_att_score: ', word_att_score)

        claim_sentence = torch.mul(claim_representation.unsqueeze(1), sentence_representations)
        rationale_score = self.rationale_linear(claim_sentence)
        # att_scores = rationale_score[:, :, 1]
        # print('rationale_score: ', rationale_score[:, :, 1])

        if bool(torch.rand(1) < rationale_sample):  # Choose sentence according to predicted rationale
            valid_scores = rationale_score[:, :, 1] > rationale_score[:, :, 0]
        else:
            valid_scores = rationale_label == 1  # Ground truth

        paragraph_representations, sen_att_score = self.sentence_attention(sentence_representations,
                                                                           sentence_mask, valid_scores)
        # print('sen_att_score: ', sen_att_score)
        # print(100 * '-*=')

        claim_paragraph = torch.mul(claim_representation, paragraph_representations)
        abstract_score = self.abstract_linear(claim_paragraph)

        abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
        rationale_pred = torch.argmax(rationale_score.cpu(), dim=-1)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
                         zip(rationale_pred, sentence_mask.bool())]

        # if bool(torch.rand(1) < rationale_sample):  # Choose sentence according to predicted rationale
        #     rationale_pred_label = torch.where(rationale_pred == 2, torch.zeros_like(rationale_pred),
        #                                        rationale_pred).float().cuda()
        # else:
        #     rationale_pred_label = rationale_label.float()  # Ground samples

        if abstract_label is not None:
            abstract_loss = self.abstract_criterion(abstract_score, abstract_label)
        else:
            abstract_loss = None

        if rationale_label is not None:
            rationale_pred_label = torch.where(rationale_pred == 2, torch.zeros_like(rationale_pred),
                                               rationale_pred).float().cuda()
            # output, target = ignore_padding(rationale_score, rationale_label)
            # rationale_loss = self.rationale_criterion(output, target.unsqueeze(1))
            rationale_loss = self.rationale_criterion(rationale_score.view(-1, self.num_rationale_label),
                                                      rationale_label.view(-1))
            sentence_loss1 = self.retrieval_rationale_criterion(sentence_att_scores.view(-1),
                                                                rationale_pred_label.view(-1).detach())
            sentence_loss2 = self.retrieval_rationale_criterion(rationale_pred_label.view(-1),
                                                                sentence_att_scores.view(-1).detach())
            bce_loss = sentence_loss1 + sentence_loss2
            # rationale_loss = rationale_loss + (alpha * sentence_loss) / 10
        else:
            rationale_loss = None
            bce_loss = None

        if retrieval_label is not None:
            retrieval_loss = self.retrieval_criterion(abstract_retrieval, retrieval_label)
            # retrieval_loss = self.retrieval_criterion(abstract_retrieval, retrieval_label.unsqueeze(1))
        else:
            retrieval_loss = None
        if train:
            return abstract_out, rationale_out, abstract_loss, rationale_loss, retrieval_loss, bce_loss
        return abstract_out, rationale_out, retrieval_out




