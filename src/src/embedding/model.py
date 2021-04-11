import torch

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from transformers import AutoModel
from embedding.loss import FocalLoss, MultiFocalLoss, DiceLoss


class WordAttention(nn.Module):
    """
    word-level attention
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super(WordAttention, self).__init__()
        self.word_attention = nn.Linear(input_size, output_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = nn.Linear(output_size, 1, bias=True)

    def forward(self, x, token_mask):
        att_s = self.dropout(x.view(-1, x.size(-1)))
        att_s = self.word_attention(att_s)
        att_s = self.dropout(torch.tanh(att_s))
        raw_att_scores = self.att_scorer(att_s).squeeze(-1).view(x.size(0), x.size(1),
                                                                 x.size(2))  # (batch_size, num_sentence, num_token)
        # raw_att_scores = torch.exp(raw_att_scores - raw_att_scores.max())  # 不加的效果更好 大约5个点
        att_scores = F.softmax(raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf')), dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)  # replace NaN with 0
        # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        batch_att_scores = att_scores.view(-1, att_scores.size(-1))  # (batch_size * num_sentence, num_token)
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
        # (batch_size * num_sentence, input_size)
        out = out.view(x.size(0), x.size(1), x.size(-1))
        mask = token_mask[:, :, 0]
        return out, mask


class SentenceAttention(nn.Module):
    """
    sentence-level attention
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super(SentenceAttention, self).__init__()
        self.sentence_attention = nn.Linear(input_size, output_size, bias=False)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = nn.Linear(output_size, 1, bias=True)
        self.contextualized = False

    def forward(self, sentence_reps, sentence_mask, valid_scores):
        # sentence_reps: [batch_size, num_sentence, hidden_dim]
        sentence_mask = torch.logical_and(sentence_mask, valid_scores)
        # Force those sentence representations in paragraph without rationale to be 0.
        # NEI_mask = (torch.sum(sentence_mask, axis=1) > 0).long().unsqueeze(-1).expand(-1, sentence_reps.size(-1))

        # if sentence_reps.size(0) > 0:
        att_s = self.sentence_attention(sentence_reps)
        u_i = self.dropout(torch.tanh(att_s))
        u_w = self.att_scorer(u_i).squeeze(-1).view(sentence_reps.size(0), sentence_reps.size(1))
        val = u_w.max()
        att_scores = torch.exp(u_w - val)
        # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        att_scores = F.softmax(att_scores.masked_fill((~sentence_mask).bool(), -1e4), dim=-1)  # [batch_size, num_sentence]
        result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)  # [batch_size, hidden_dim]
        return result
        # return sentence_reps[:, 0, :]


class SentenceClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks (CNN model). """

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(SentenceClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)
        self.num_channels = hidden_size // 3
        self.kernel_size = [2, 3, 4]
        self.max_position_embeddings = 512

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
                               kernel_size=(self.kernel_size[0], hidden_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
                               kernel_size=(self.kernel_size[1], hidden_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
                               kernel_size=(self.kernel_size[2], hidden_size))
        # self.conv4 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
        #                        kernel_size=(self.kernel_size[3], hidden_size))

        self.attention_size = 16
        self.w_omega = Variable(torch.zeros(hidden_size, self.attention_size).cuda())
        self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        # self.apply(self.init_bert_weights)

    def conv_pooling(self, x, conv):
        out = conv(x)
        activation = F.relu(out.squeeze(3))
        out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return out

    def forward(self, x):
        pooled_output = x.unsqueeze(1)

        h1 = self.conv_pooling(pooled_output, self.conv1)
        h2 = self.conv_pooling(pooled_output, self.conv2)
        h3 = self.conv_pooling(pooled_output, self.conv3)
        # h4 = self.conv_pooling(pooled_output, self.conv4)
        pooled_output = torch.cat([h1, h2, h3], 1)
        pooled_output = self.dropout(pooled_output)
        # print(pooled_output.size())
        # output = self.attention(pooled_output)
        output = self.classifier(pooled_output)
        return output


class BiLstmAttention(nn.Module):
    def __init__(self, args, hidden_size, num_labels, dropout=0.1):
        super(BiLstmAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = num_labels
        self.dropout = dropout
        self.embed_dim = args.embed_size
        self.vocab_size = args.vocab_size
        self.bidirectional = True
        self.batch_size = args.batch_size_gpu
        self.sequence_length = 512
        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = 16
        self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
        self.u_omega = Variable(torch.zeros(self.attention_size).cuda())

        self.label = nn.Linear(hidden_size * self.layer_size, self.output_size)

    def attention(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        att_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        att_hidden_layer = torch.mm(att_tanh, torch.Tensor.reshape(self.u_omega, [-1, -1]))
        exps = torch.Tensor.reshape(torch.exp(att_hidden_layer), [-1, self.sequence_length])
        alpha = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, -1])
        alpha = torch.Tensor.reshape(alpha, [-1, self.sequence_length, 1])
        state = lstm_output.permute(1, 0, 2)
        att_output = torch.sum(state * alpha, 1)
        return att_output

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        lstm_output, (_, _) = self.lstm(x, (h_0, c_0))
        att_output = self.attention(lstm_output)
        logits = self.label(att_output)
        return logits


class ClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks. """

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, num_labels, bias=True)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


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


class JointModelClassifier(nn.Module):
    def __init__(self, args):
        super(JointModelClassifier, self).__init__()
        self.num_abstract_label = 3
        self.num_rationale_label = 2
        self.sim_label = 2
        self.bert = AutoModel.from_pretrained(args.model)
        # self.abstract_criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0.1, alpha=0.01, reduction='mean',
        #                                             square_denominator=True)
        # self.abstract_criterion = MultiFocalLoss(3, alpha=[0.1, 0.6, 0.3])
        self.abstract_criterion = nn.CrossEntropyLoss()
        # self.rationale_criterion = nn.CrossEntropyLoss(ignore_index=2)
        # self.rationale_criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
        self.rationale_criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0.1, alpha=0.01, reduction='mean',
                                            square_denominator=True)
        self.similarity_criterion = nn.CrossEntropyLoss()
        self.dropout = args.dropout
        self.hidden_dim = args.hidden_dim
        self.word_attention = WordAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.sentence_attention = SentenceAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.rationale_linear = ClassificationHead(self.hidden_dim, self.num_rationale_label, dropout=self.dropout)
        # self.abstract_linear = ClassificationHead(self.hidden_dim, self.num_abstract_label, dropout=self.dropout)
        # self.similarity_linear = SentenceClassificationHead(self.hidden_dim, self.sim_label, dropout=self.dropout)
        self.abstract_linear = SentenceClassificationHead(self.hidden_dim,
                                                          self.num_abstract_label, dropout=self.dropout)
        self.similarity_linear = ClassificationHead(self.hidden_dim, self.sim_label, dropout=self.dropout)

        self.rationale_module = [
            self.word_attention,
            self.rationale_linear,
            self.rationale_criterion,
        ]

        self.label_module = [
            self.abstract_criterion,
            self.abstract_linear,
        ]

        self.similarity_module = [
            self.sentence_attention,
            self.similarity_linear,
            self.similarity_criterion,
        ]

    def forward(self, encoded_dict, transformation_indices, abstract_label=None, rationale_label=None, sim_label=None,
                train=False):

        batch_indices, indices_by_batch, mask = transformation_indices
        # (batch_size, num_sep, num_token)
        bert_out = self.bert(**encoded_dict)[0]  # [batch_size, sequence_len, hidden_dim]
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]  # get SciBert tokens
        # bert_tokens: [batch_size, num_sentence, num_token, hidden_dim]
        sentence_representations, sentence_mask = self.word_attention(bert_tokens, mask)
        # print(sentence_representations.size())
        rationale_score = self.rationale_linear(sentence_representations)
        # word_attention_scores = rationale_score[:, :, 1]
        valid_scores = rationale_score[:, :, 1] > rationale_score[:, :, 0]
        paragraph_representations = self.sentence_attention(sentence_representations, sentence_mask, valid_scores)
        # abstract_reps = self.bert_sentence_attention(encoded_dict['input_ids'], doc_length, sentence_length,
        #                                              encoded_dict['attention_mask'], encoded_dict['token_type_ids'],
        #                                              bert_out)
        # abstract_score = self.abstract_linear(abstract_reps)
        # abstract_score = self.abstract_linear(paragraph_representations)  # attention + linear
        # bert_embedding = self.bert(**encoded)[0]
        abstract_score = self.abstract_linear(bert_out)  # CNN
        sim_score = self.similarity_linear(paragraph_representations)

        abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
        rationale_pred = torch.argmax(rationale_score.cpu(), dim=-1)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
                         zip(rationale_pred, sentence_mask.bool())]
        # sim_out = torch.argmax(sim_score.cpu(), dim=-1).detach().numpy().tolist()

        if abstract_label is not None:
            abstract_loss = self.abstract_criterion(abstract_score, abstract_label)
        else:
            abstract_loss = None

        if rationale_label is not None:
            # rationale_loss = self.rationale_criterion(rationale_score.view(-1, self.num_rationale_label),
            #                                           rationale_label.view(-1))
            output, target = ignore_padding(rationale_score, rationale_label)
            # rationale_loss = self.rationale_criterion(output.view(-1, 2), target.view(-1))
            # print(target)
            rationale_loss = self.rationale_criterion(output, target)
        else:
            rationale_loss = None

        if sim_label is not None:
            sim_loss = self.similarity_criterion(sim_score, sim_label)
        else:
            sim_loss = None
        if train:
            return abstract_out, rationale_out, abstract_loss, rationale_loss, sim_loss
        return abstract_out, rationale_out


# class JointModelClassifier(nn.Module):
#     def __init__(self, args):
#         super(JointModelClassifier, self).__init__()
#         self.num_abstract_label = 3
#         self.num_rationale_label = 2
#         self.sim_label = 2
#         self.bert = AutoModel.from_pretrained(args.model)
#         self.abstract_criterion = DiceLoss(with_logits=True, smooth=0.5, ohem_ratio=0.1, alpha=0.01, reduction='mean',
#                                            square_denominator=True)
#         self.rationale_criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0.1, alpha=0.01, reduction='mean',
#                                             square_denominator=True)
#         self.retrieval_criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0.1, alpha=0.01, reduction='mean',
#                                             square_denominator=True)
#         self.dropout = args.dropout
#         self.hidden_dim = args.hidden_dim
#         self.word_attention = WordAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
#         self.sentence_attention = SentenceAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
#         self.rationale_linear = ClassificationHead(self.hidden_dim, self.num_rationale_label, dropout=self.dropout)
#         self.abstract_linear = ClassificationHead(self.hidden_dim, self.num_abstract_label, dropout=self.dropout)
#         self.abstract_retrieval = AttentionCNN(self.hidden_dim, self.sim_label, dropout=self.dropout)
#         self.conv_layer = ConvLayer(self.hidden_dim, dropout=self.dropout)
#
#     def forward(self, encoded_dict, transformation_indices, abstract_label=None, rationale_label=None,
#                 sim_label=None, train=False, retrieval_out=False):
#         batch_indices, indices_by_batch, mask = transformation_indices
#         # (batch_size, num_sep, num_token)
#         output = self.bert(**encoded_dict)  # [batch_size, sequence_len, hidden_dim]
#         bert_out = output[0]
#         cnn_out = self.conv_layer(bert_out)
#         # classifier_out = output[1]  # [batch_size, hidden_dim]
#         bert_tokens = bert_out[batch_indices, indices_by_batch, :]  # get Bert tokens(sentences level)
#         # bert_tokens: [batch_size, num_sentence, num_token, hidden_dim]
#         # match_batch_indices, match_indices_by_batch, match_mask = match_indices
#         # x1 = bert_out[match_batch_indices, match_indices_by_batch, :][:, 0, :, :].unsqueeze(1)
#         # x2 = bert_out[match_batch_indices, match_indices_by_batch, :][:, 1, :, :].unsqueeze(1)
#         # abstract_retrieval = self.Abcnn(x1, x2)
#
#         abstract_retrieval = self.abstract_retrieval(bert_out, cnn_out)
#         # valid_abstract = abstract_retrieval[:, 1] > abstract_retrieval[:, 0]
#         sentence_representations, sentence_mask, word_att_score = self.word_attention(bert_tokens, mask)
#         # rationale_score = self.rationale_linear(sentence_representations)
#         # sentence_reps = sentence_representations.permute(1, 0, 2)
#         # rationale_score = self.rationale_linear(sentence_reps[:, 0, :]).unsqueeze(0)
#         # for i in range(1, sentence_reps.size(1), 1):
#         #     score = self.rationale_linear(sentence_reps[:, i, :]).unsqueeze(0)
#         #     rationale_score = torch.cat([rationale_score, score], 0)
#         # # sentence_attention_scores = rationale_score[:, :, 1]
#         # valid_scores = rationale_score[:, :, 1] > rationale_score[:, :, 0]
#         paragraph_representations, sen_att_score = self.sentence_attention(sentence_representations,
#                                                                            sentence_mask)
#         rationale_score = self.rationale_linear(sentence_representations)
#         # abstract_reps = self.bert_sentence_attention(encoded_dict['input_ids'], doc_length, sentence_length,
#         #                                              encoded_dict['attention_mask'], encoded_dict['token_type_ids'],
#         #                                              bert_out)
#         # abstract_score = self.abstract_linear(abstract_reps)
#         # abstract_score = self.abstract_linear(paragraph_representations)  # attention + linear
#         # bert_embedding = self.bert(**encoded)[0]
#         # rationale_score = self.rationale_linear(sentence_reps)
#
#         # sim_score = self.similarity_linear(paragraph_representations)
#         # abstract_mask, abstract_batch_indices = get_index(mask)
#         # x = torch.mul(bert_tokens, sen_att_score.unsqueeze(2).unsqueeze(2))
#         # x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))[abstract_batch_indices, abstract_mask, :]
#         # x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
#         # sim_score = self.similarity_linear(x)
#
#         abstract_score = self.abstract_linear(paragraph_representations)
#         # abstract_linear_score = sim_score[:, [1, 0, 1]] / 2
#         # abstract_linear_score = self.abstract_linear(paragraph_representations)
#         # abstract_score = (abstract_score + abstract_linear_score)
#
#         abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
#         rationale_pred = torch.argmax(rationale_score.cpu(), dim=-1)
#         rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
#                          zip(rationale_pred, sentence_mask.bool())]
#         # sim_out = torch.argmax(sim_score.cpu(), dim=-1).detach().numpy().tolist()
#
#         if abstract_label is not None:
#             abstract_loss = self.abstract_criterion(abstract_score, abstract_label)
#         else:
#             abstract_loss = None
#
#         if rationale_label is not None:
#             # rationale_loss = self.rationale_criterion(rationale_score.view(-1, self.num_rationale_label),
#             #                                           rationale_label.view(-1))
#             output, target = ignore_padding(rationale_score, rationale_label)
#             # rationale_loss = self.rationale_criterion(output.view(-1, 2), target.view(-1))
#             rationale_loss = self.rationale_criterion(output, target)
#         else:
#             rationale_loss = None
#
#         if sim_label is not None:
#             retrieval_loss = self.retrieval_criterion(abstract_retrieval, sim_label)
#         else:
#             retrieval_loss = None
#         if train:
#             return abstract_out, rationale_out, abstract_loss, rationale_loss, retrieval_loss
#         if retrieval_out:
#             return torch.argmax(abstract_retrieval.cpu(), dim=-1).detach().numpy().tolist()
#         return abstract_out, rationale_out


'''
github  3.28
'''
# import torch
#
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional
#
# from torch.autograd import Variable
# from torch.nn.modules.loss import _WeightedLoss
# from transformers import AutoModel
# from embedding.loss import FocalLoss, MultiFocalLoss
#
#
# class WordAttention(nn.Module):
#     """
#     word-level attention
#     """
#     def __init__(self, input_size, output_size, dropout=0.1):
#         super(WordAttention, self).__init__()
#         self.word_attention = nn.Linear(input_size, output_size, bias=True)
#         self.dropout = nn.Dropout(dropout)
#         self.att_scorer = nn.Linear(output_size, 1, bias=True)
#
#     def forward(self, x, token_mask):
#         att_s = self.dropout(x.view(-1, x.size(-1)))
#         att_s = self.word_attention(att_s)
#         att_s = self.dropout(torch.tanh(att_s))
#         raw_att_scores = self.att_scorer(att_s).squeeze(-1).view(x.size(0), x.size(1),
#                                                                  x.size(2))  # (batch_size, num_sentence, num_token)
#         raw_att_scores = torch.exp(raw_att_scores - raw_att_scores.max())
#         att_scores = F.softmax(raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf')), dim=-1)
#         att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
#                                  att_scores)  # replace NaN with 0
#         # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
#         batch_att_scores = att_scores.view(-1, att_scores.size(-1))  # (batch_size * num_sentence, num_token)
#         out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
#         # (batch_size * num_sentence, input_size)
#         out = out.view(x.size(0), x.size(1), x.size(-1))
#         mask = token_mask[:, :, 0]
#         return out, mask
#
#
# class SentenceAttention(nn.Module):
#     """
#     sentence-level attention
#     """
#     def __init__(self, input_size, output_size, dropout=0.1):
#         super(SentenceAttention, self).__init__()
#         self.sentence_attention = nn.Linear(input_size, output_size, bias=False)
#         self.activation = torch.tanh
#         self.dropout = nn.Dropout(dropout)
#         self.att_scorer = nn.Linear(output_size, 1, bias=True)
#         self.contextualized = False
#
#     def forward(self, sentence_reps, sentence_mask, valid_scores):
#         sentence_mask = torch.logical_and(sentence_mask, valid_scores)
#
#         if sentence_reps.size(0) > 0:
#             att_s = self.sentence_attention(sentence_reps)
#             u_i = self.dropout(torch.tanh(att_s))
#             u_w = self.att_scorer(u_i).squeeze(-1).view(sentence_reps.size(0), sentence_reps.size(1))
#             val = u_w.max()
#             att_scores = torch.exp(u_w - val)
#             # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
#             att_scores = F.softmax(att_scores.masked_fill((~sentence_mask).bool(), -1e4), dim=-1)
#             result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
#             return result  # * NEI_mask
#         return sentence_reps[:, 0, :]
#
#
# class SentenceClassificationHead(nn.Module):
#     """ Head for sentence-level classification tasks (CNN model). """
#
#     def __init__(self, hidden_size, num_labels, dropout=0.1):
#         super(SentenceClassificationHead, self).__init__()
#         self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Linear(hidden_size, num_labels, bias=True)
#         self.num_channels = hidden_size // 3
#         self.kernel_size = [2, 3, 4]
#         self.max_position_embeddings = 512
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
#                                kernel_size=(self.kernel_size[0], hidden_size))
#         self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
#                                kernel_size=(self.kernel_size[1], hidden_size))
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels,
#                                kernel_size=(self.kernel_size[2], hidden_size))
#
#         self.pool1 = nn.MaxPool1d(kernel_size=self.max_position_embeddings - 2 + 1)
#         self.pool2 = nn.MaxPool1d(kernel_size=self.max_position_embeddings - 3 + 1)
#         self.pool3 = nn.MaxPool1d(kernel_size=self.max_position_embeddings - 4 + 1)
#
#         self.attention_size = 16
#         self.w_omega = Variable(torch.zeros(hidden_size, self.attention_size).cuda())
#         self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
#         # self.apply(self.init_bert_weights)
#
#     def conv_pooling(self, x, conv):
#         out = conv(x)
#         activation = F.relu(out.squeeze(3))
#         out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
#         return out
#
#     def forward(self, x):
#         pooled_output = x.unsqueeze(1)
#
#         h1 = self.conv_pooling(pooled_output, self.conv1)
#         h2 = self.conv_pooling(pooled_output, self.conv2)
#         h3 = self.conv_pooling(pooled_output, self.conv3)
#         pooled_output = torch.cat([h1, h2, h3], 1)
#         pooled_output = self.dropout(pooled_output)
#         # print(pooled_output.size())
#         # output = self.attention(pooled_output)
#         output = self.classifier(pooled_output)
#         return output
#
#
# class BiLstmAttention(nn.Module):
#     def __init__(self, args, hidden_size, num_labels, dropout=0.1):
#         super(BiLstmAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = num_labels
#         self.dropout = dropout
#         self.embed_dim = args.embed_size
#         self.vocab_size = args.vocab_size
#         self.bidirectional = True
#         self.batch_size = args.batch_size_gpu
#         self.sequence_length = 512
#         self.layer_size = 1
#         self.lstm = nn.LSTM(self.embed_dim,
#                             self.hidden_size,
#                             self.layer_size,
#                             dropout=self.dropout,
#                             bidirectional=self.bidirectional)
#         if self.bidirectional:
#             self.layer_size = self.layer_size * 2
#         else:
#             self.layer_size = self.layer_size
#
#         self.attention_size = 16
#         self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
#         self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
#
#         self.label = nn.Linear(hidden_size * self.layer_size, self.output_size)
#
#     def attention(self, lstm_output):
#         output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
#         att_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
#         att_hidden_layer = torch.mm(att_tanh, torch.Tensor.reshape(self.u_omega, [-1, -1]))
#         exps = torch.Tensor.reshape(torch.exp(att_hidden_layer), [-1, self.sequence_length])
#         alpha = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, -1])
#         alpha = torch.Tensor.reshape(alpha, [-1, self.sequence_length, 1])
#         state = lstm_output.permute(1, 0, 2)
#         att_output = torch.sum(state * alpha, 1)
#         return att_output
#
#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
#         c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
#         lstm_output, (_, _) = self.lstm(x, (h_0, c_0))
#         att_output = self.attention(lstm_output)
#         logits = self.label(att_output)
#         return logits
#
#
# class ClassificationHead(nn.Module):
#     """ Head for sentence-level classification tasks. """
#
#     def __init__(self, hidden_size, num_labels, dropout=0.1):
#         super().__init__()
#         self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
#         self.dropout = nn.Dropout(dropout)
#         self.output = nn.Linear(hidden_size, num_labels, bias=True)
#
#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.output(x)
#         return x
#
#
# def ignore_padding(output, target, padding_idx=2):
#     """ Remove padded sentences and label(2). """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     target = target.view(-1)
#     output = output.view(-1, 2)
#     num_padding = 0
#     n = len(target)
#     for i in range(n - 1, -1, -1):
#         if target[i] == padding_idx:
#             num_padding += 1
#     target = target[:n - num_padding]
#     output = output[:n - num_padding]
#     return output.to(device), target.to(device)
#
#
# class JointModelClassifier(nn.Module):
#     def __init__(self, args):
#         super(JointModelClassifier, self).__init__()
#         self.num_abstract_label = 3
#         self.num_rationale_label = 2
#         self.sim_label = 2
#         self.bert = AutoModel.from_pretrained(args.model)
#         self.abstract_criterion = nn.CrossEntropyLoss()
#         # self.abstract_criterion = MultiFocalLoss(3, alpha=[0.1, 0.7, 0.2])
#         # self.rationale_criterion = nn.CrossEntropyLoss(ignore_index=2)
#         self.rationale_criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
#         self.similarity_criterion = nn.CrossEntropyLoss()
#         self.dropout = args.dropout
#         self.hidden_dim = args.hidden_dim
#         self.word_attention = WordAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
#         self.sentence_attention = SentenceAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
#         self.rationale_linear = ClassificationHead(self.hidden_dim, self.num_rationale_label, dropout=self.dropout)
#         self.abstract_linear = SentenceClassificationHead(self.hidden_dim,
#                                                           self.num_abstract_label, dropout=self.dropout)
#         self.similarity_linear = ClassificationHead(self.hidden_dim, self.sim_label, dropout=self.dropout)
#
#         self.rationale_module = [
#             self.word_attention,
#             self.rationale_linear,
#             self.rationale_criterion,
#         ]
#
#         self.label_module = [
#             self.abstract_criterion,
#             self.abstract_linear,
#         ]
#
#         self.similarity_module = [
#             self.sentence_attention,
#             self.similarity_linear,
#             self.similarity_criterion,
#         ]
#
#     def forward(self, encoded_dict, transformation_indices, abstract_label=None, rationale_label=None, sim_label=None,
#                 train=False):
#
#         batch_indices, indices_by_batch, mask = transformation_indices
#         # (batch_size, num_sep, num_token)
#         bert_out = self.bert(**encoded_dict)[0]  # (BATCH_SIZE, sequence_len, BERT_DIM)
#         bert_tokens = bert_out[batch_indices, indices_by_batch, :]  # get SciBert tokens
#
#         # bert_tokens: (batch_size, num_sep, num_token, BERT_dim)
#         sentence_representations, sentence_mask = self.word_attention(bert_tokens, mask)
#         # print(sentence_representations.size())
#         rationale_score = self.rationale_linear(sentence_representations)
#         # word_attention_scores = rationale_score[:, :, 1]
#         valid_scores = rationale_score[:, :, 1] > rationale_score[:, :, 0]
#         paragraph_representations = self.sentence_attention(sentence_representations, sentence_mask, valid_scores)
#         # abstract_reps = self.bert_sentence_attention(encoded_dict['input_ids'], doc_length, sentence_length,
#         #                                              encoded_dict['attention_mask'], encoded_dict['token_type_ids'],
#         #                                              bert_out)
#         # abstract_score = self.abstract_linear(abstract_reps)
#         # abstract_score = self.abstract_linear(paragraph_representations)  # attention + linear
#         # bert_embedding = self.bert(**encoded)[0]
#         abstract_score = self.abstract_linear(bert_out)  # CNN
#         sim_score = self.similarity_linear(paragraph_representations)
#
#         abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
#         rationale_pred = torch.argmax(rationale_score.cpu(), dim=-1)
#         rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
#                          zip(rationale_pred, sentence_mask.bool())]
#         # sim_out = torch.argmax(sim_score.cpu(), dim=-1).detach().numpy().tolist()
#
#         if abstract_label is not None:
#             abstract_loss = self.abstract_criterion(abstract_score, abstract_label)
#         else:
#             abstract_loss = None
#
#         if rationale_label is not None:
#             # rationale_loss = self.rationale_criterion(rationale_score.view(-1, self.num_rationale_label),
#             #                                           rationale_label.view(-1))
#             output, target = ignore_padding(rationale_score, rationale_label)
#             rationale_loss = self.rationale_criterion(output.view(-1, 2), target.view(-1).unsqueeze(1))
#         else:
#             rationale_loss = None
#
#         if sim_label is not None:
#             sim_loss = self.similarity_criterion(sim_score, sim_label)
#         else:
#             sim_loss = None
#         if train:
#             return abstract_out, rationale_out, abstract_loss, rationale_loss, sim_loss
#         return abstract_out, rationale_out
