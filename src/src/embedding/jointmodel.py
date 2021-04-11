import torch

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from torch.autograd import Variable
from transformers import AutoModel
from embedding.loss import FocalLoss, MultiFocalLoss, DiceLoss
from embedding.ABCNN_github import ABCNN, weights_init


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
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
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
    word-level attention
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super(WordAttention, self).__init__()
        self.word_attention = nn.Linear(input_size, output_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = nn.Linear(output_size, 1, bias=True)

    def forward(self, x, token_mask, valid_abstract):
        # batch_indices, indices_by_batch, token_mask = transformation_indices
        # x = self.Lstm(x).permute(1, 0, 2)  # [batch_indices, sequence_len, hidden_dim]
        # x = x[batch_indices, indices_by_batch, :]  # [batch_size, num_sentence, num_token, hidden_dim]
        valid_abstract = valid_abstract.repeat(token_mask.shape[1], 1).transpose(0, 1).unsqueeze(2)
        att_s = self.dropout(x.view(-1, x.size(-1)))
        att_s = self.word_attention(att_s)
        att_s = self.dropout(torch.tanh(att_s))  # [batch_size * num_sentence * num_token, hidden_dim]
        raw_att_scores = self.att_scorer(att_s).squeeze(-1).view(x.size(0), x.size(1),
                                                                 x.size(2))  # [batch_size, num_sentence, num_token]
        # raw_att_scores = torch.exp(raw_att_scores - raw_att_scores.max())  #
        word_mask = torch.logical_and((1 - token_mask).bool(), ~valid_abstract)
        # word_mask = (1 - token_mask).bool()
        att_scores = F.softmax(raw_att_scores.masked_fill(word_mask, float('-inf')), dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)  # replace NaN with 0
        # att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        # batch_att_scores: word attention scores. [batch_size * num_sentence, num_token]
        batch_att_scores = att_scores.view(-1, att_scores.size(-1))  # word attention weight matrix.
        # out:  # sentence_representations. [batch_size, num_sentence, hidden_dim]
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
        out = out.view(x.size(0), x.size(1), x.size(-1))  # [batch_size, num_sentence, hidden_dim]
        mask = token_mask[:, :, 0]  # [batch_size, num_sentence]
        return out, mask, att_scores


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
        self.hidden_size = input_size // 2
        self.lstm = nn.LSTM(input_size, self.hidden_size, 1, dropout=dropout, bidirectional=True)

    def forward(self, sentence_reps, sentence_mask, valid_scores):
        # valid_abstract = valid_abstract.repeat(sentence_mask.shape[1], 1).transpose(0, 1)
        sentence_mask = torch.logical_and(sentence_mask, valid_scores)
        # Force those sentence representations in paragraph without rationale to be 0.
        # NEI_mask = (torch.sum(sentence_mask, axis=1) > 0).long().unsqueeze(-1).expand(-1, sentence_reps.size(-1))
        h_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size).cuda())
        # print(sentence_reps.shape)
        sent_embeddings = self.dropout(sentence_reps)
        sentence_embedding, (_, _) = self.lstm(sent_embeddings, (h_0, c_0))
        # if sentence_reps.size(0) > 0:
        att_s = self.sentence_attention(sentence_embedding)  # [batch_size, num_sentence, hidden_size,]
        u_i = self.dropout(torch.tanh(att_s))  # u_i = tanh(W_s * h_i + b). [batch_size, num_sentence, hidden_size,]
        u_w = self.att_scorer(u_i).squeeze(-1).view(sentence_reps.size(0), sentence_reps.size(1))  # [batch_size, num_sentence]
        # sentence_score: sentence attention weight matrix.
        # sentence_score = torch.exp(u_w - u_w.max()) / torch.sum(torch.exp(u_w - u_w.max()), dim=1, keepdim=True)
        u_w = u_w.masked_fill((~sentence_mask).bool(), -1e4)
        val = u_w.max()
        # att_scores: sentence attention scores. [batch_size, num_sentence]
        att_scores = torch.exp(u_w - val)
        att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        # att_scores: sentence attention scores. [batch_size, num_sentence]
        # att_scores = att_scores.masked_fill((~sentence_mask).bool(), -1e4)
        # att_scores = F.softmax(att_scores, dim=-1)
        # print(att_scores, sentence_mask)
        # result: abstract representations. [batch_size, hidden_dim]
        result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
        # sentence_reps = torch.mul(sentence_reps, att_scores.unsqueeze(2))
        return result, att_scores
        # return sentence_reps[:, 0, :], sentence_reps


class AttentionCNN(nn.Module):
    """ Head for sentence-level classification tasks (CNN model). """
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(AttentionCNN, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = ClassificationHead(hidden_size, num_labels, dropout=dropout)
        self.hidden_size = hidden_size
        self.conv_layer = ConvLayer(self.hidden_size, dropout=dropout)

        self.embed_dim = hidden_size
        self.bidirectional = True
        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size // 2,
                            self.layer_size,
                            dropout=dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = 16
        self.w_omega = Variable(torch.zeros(hidden_size, self.attention_size).cuda())
        self.u_omega = Variable(torch.zeros(self.attention_size).cuda())

    def attention_pooling(self, x1, x2):
        # x1: cnn out, x2: lstm out
        att_e = self.cosine_similarity(x1, x2)
        att_scores = torch.exp(att_e)
        att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)
        att_scores = torch.Tensor.reshape(att_scores, [1, x1.shape[0], -1])
        x = x1.permute(1, 0, 2)
        att_output = torch.sum(x * att_scores, 1)
        # output_reshape = torch.Tensor.reshape(x, [-1, self.hidden_size])
        # att_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # att_hidden_layer = torch.mm(att_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # exps = torch.Tensor.reshape(torch.exp(att_hidden_layer), [-1, x.shape[0]])
        # alpha = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # alpha = torch.Tensor.reshape(alpha, [-1, x.shape[0], 1])
        # state = x.permute(1, 0, 2)
        # att_output = torch.sum(state * alpha, 1)
        return att_output

    def forward(self, x, cnn_out):
        cnn_out = cnn_out.permute(1, 0, 2)
        x = x.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(self.layer_size, x.size(1), self.hidden_size // 2).cuda())
        c_0 = Variable(torch.zeros(self.layer_size, x.size(1), self.hidden_size // 2).cuda())
        lstm_output, (_, _) = self.lstm(x, (h_0, c_0))
        att_out = self.attention_pooling(cnn_out, lstm_output)
        output = self.classifier(att_out)
        return output

    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2).unsqueeze(1)


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
        self.abstract_criterion = DiceLoss(with_logits=True, smooth=0.5, ohem_ratio=0.1, alpha=0.01, reduction='mean',
                                           square_denominator=True)
        self.rationale_criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0.1, alpha=0.01, reduction='mean',
                                            square_denominator=True)
        self.retrieval_criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0.1, alpha=0.01, reduction='mean',
                                            square_denominator=True)
        self.dropout = args.dropout
        self.hidden_dim = args.hidden_dim
        self.word_attention = WordAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.sentence_attention = SentenceAttention(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
        self.rationale_linear = ClassificationHead(self.hidden_dim, self.num_rationale_label, dropout=self.dropout)
        self.abstract_linear = ClassificationHead(self.hidden_dim, self.num_abstract_label, dropout=self.dropout)
        self.abstract_retrieval = AttentionCNN(self.hidden_dim, self.sim_label, dropout=self.dropout)
        self.conv_layer = ConvLayer(self.hidden_dim, dropout=self.dropout)

    def forward(self, encoded_dict, transformation_indices, match_indices, abstract_label=None, rationale_label=None,
                sim_label=None, train=False, retrieval_out=False, sample_p=1):
        batch_indices, indices_by_batch, mask = transformation_indices
        match_batch_indices, match_indices_by_batch, match_mask = match_indices
        # (batch_size, num_sep, num_token)
        output = self.bert(**encoded_dict)  # [batch_size, sequence_len, hidden_dim]
        bert_out = output[0]

        corpus_tokens = bert_out[match_batch_indices, match_indices_by_batch, :]
        corpus_tokens = corpus_tokens.squeeze(1)  # [batch_size, sequence_len, hidden_dim]
        cnn_out = self.conv_layer(corpus_tokens)

        # classifier_out = output[1]  # [batch_size, hidden_dim]
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]  # get Bert tokens(sentences level)
        # bert_tokens: [batch_size, num_sentence, num_token, hidden_dim]
        abstract_retrieval = self.abstract_retrieval(corpus_tokens, cnn_out)

        if bool(torch.rand(1) < sample_p):  # Choose abstract according to predicted abstract
            valid_abstract = abstract_retrieval[:, 1] > abstract_retrieval[:, 0]
        else:
            valid_abstract = sim_label == 1  # Ground truth

        sentence_representations, sentence_mask, word_att_score = self.word_attention(bert_tokens, mask, valid_abstract)

        rationale_score = self.rationale_linear(sentence_representations)

        if bool(torch.rand(1) < sample_p):  # Choose sentence according to predicted rationale
            valid_scores = rationale_score[:, :, 1] > rationale_score[:, :, 0]
        else:
            valid_scores = rationale_label == 1  # Ground truth

        paragraph_representations, sen_att_score = self.sentence_attention(sentence_representations,
                                                                           sentence_mask, valid_scores)

        abstract_score = self.abstract_linear(paragraph_representations)

        abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
        rationale_pred = torch.argmax(rationale_score.cpu(), dim=-1)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
                         zip(rationale_pred, sentence_mask.bool())]

        if abstract_label is not None:
            abstract_loss = self.abstract_criterion(abstract_score, abstract_label)
        else:
            abstract_loss = None

        if rationale_label is not None:
            output, target = ignore_padding(rationale_score, rationale_label)
            rationale_loss = self.rationale_criterion(output, target)
        else:
            rationale_loss = None

        if sim_label is not None:
            retrieval_loss = self.retrieval_criterion(abstract_retrieval, sim_label)
        else:
            retrieval_loss = None
        if train:
            return abstract_out, rationale_out, abstract_loss, rationale_loss, retrieval_loss
        if retrieval_out:
            return torch.argmax(abstract_retrieval.cpu(), dim=-1).detach().numpy().tolist()
        return abstract_out, rationale_out





