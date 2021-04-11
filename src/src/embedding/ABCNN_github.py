import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ABCNN(nn.Module):
    def __init__(self, hidden_dim, filter_width=1, layer_size=2, match='cosine'):
        super(ABCNN, self).__init__()
        self.layer_size = layer_size
        if match == 'cosine':
            self.distance = self.cosine_similarity
        else:
            self.distance = self.manhattan_distance

        self.sentence_length = 512
        self.num_channels = 768 // 3

        self.abcnn1 = nn.ModuleList()
        self.abcnn2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList([ApLayer(hidden_dim)])
        self.fc = nn.Linear(layer_size + 1, 2)

        for i in range(layer_size):
            self.abcnn1.append(Abcnn1Portion(self.sentence_length, hidden_dim if i == 0 else self.num_channels))
            self.abcnn2.append(Abcnn2Portion(self.sentence_length, filter_width))
            self.conv.append(
                ConvLayer(2, filter_width, hidden_dim if i == 0 else self.num_channels, self.num_channels))
            self.ap.append(ApLayer(self.num_channels))

    def forward(self, x1, x2):
        sim = []
        sim.append(self.distance(self.ap[0](x1), self.ap[0](x2)))

        for i in range(self.layer_size):
            x1, x2 = self.abcnn1[i](x1, x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.distance(self.ap[i + 1](x1), self.ap[i + 1](x2)))
            x1, x2 = self.abcnn2[i](x1, x2)

        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output

    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2).unsqueeze(1)

    def manhattan_distance(self, x1, x2):
        return torch.div(torch.norm((x1 - x2), 1, 1, keepdim=True), x1.size()[1])


class Abcnn1Portion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Abcnn1Portion, self).__init__()
        self.batchNorm = nn.BatchNorm2d(2)
        self.attention_feature_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x1, x2):
        attention_m = attention_matrix(x1, x2)

        x1_attention = self.attention_feature_layer(attention_m.permute(0, 2, 1))
        x1_attention = x1_attention.unsqueeze(1)
        x1 = torch.cat([x1, x1_attention], 1)

        x2_attention = self.attention_feature_layer(attention_m)
        x2_attention = x2_attention.unsqueeze(1)
        x2 = torch.cat([x2, x2_attention], 1)

        x1 = self.batchNorm(x1)
        x2 = self.batchNorm(x2)

        return (x1, x2)


class Abcnn2Portion(nn.Module):
    def __init__(self, sentence_length, filter_width):
        super(Abcnn2Portion, self).__init__()
        self.wp = WpLayer(sentence_length, filter_width, True)

    def forward(self, x1, x2):
        attention_m = attention_matrix(x1, x2)
        x1_a_conv = attention_m.sum(dim=1)
        x2_a_conv = attention_m.sum(dim=2)
        x1 = self.wp(x1, x1_a_conv)
        x2 = self.wp(x2, x2_a_conv)

        return (x1, x2)


class ConvLayer(nn.Module):
    def __init__(self, in_channel, filter_width, filter_height, filter_channel):
        super(ConvLayer, self).__init__()
        self.conv_1 = convolution(in_channel, filter_width, filter_height,
                                  int(filter_channel / 3) + filter_channel - 3 * int(filter_channel / 3),
                                  filter_width - 1)
        self.conv_2 = convolution(in_channel, filter_width + 4, filter_height, int(filter_channel / 3),
                                  filter_width + 1)
        self.conv_3 = convolution(in_channel, filter_width + 8, filter_height, int(filter_channel / 3),
                                  int((filter_width + 8 + filter_width - 2) / 2))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(x)
        out_3 = self.conv_3(x)
        output = torch.cat([out_1, out_2, out_3], dim=1)
        output = output.permute(0, 3, 2, 1)
        return output


def convolution(in_channel, filter_width, filter_height, filter_channel, padding):
    '''convolution layer
    '''
    model = nn.Sequential(
        nn.Conv2d(in_channel, filter_channel, (filter_width, filter_height), stride=1, padding=(padding, 0)),
        nn.BatchNorm2d(filter_channel),
        nn.Tanh()
    )
    return model


def attention_matrix(x1, x2, eps=1e-6):
    '''compute attention matrix using match score

    1 / (1 + |x · y|)
    |·| is euclidean distance
    Parameters
    ----------
    x1, x2 : 4-D torch Tensor
        size (batch_size, 1, sentence_length, width)

    Returns
    -------
    output : 3-D torch Tensor
        match score result of size (batch_size, sentence_length(for x2), sentence_length(for x1))
    '''
    eps = torch.tensor(eps)
    one = torch.tensor(1.)
    euclidean = (torch.pow(x1 - x2.permute(0, 2, 1, 3), 2).sum(dim=3) + eps).sqrt()
    return (euclidean + one).reciprocal()


class ApLayer(nn.Module):
    '''column-wise averaging over all columns
    '''

    def __init__(self, width):
        super(ApLayer, self).__init__()
        self.ap = nn.AvgPool2d((1, width), stride=1)

    def forward(self, x):
        '''
        1. average pooling
            x size (batch_size, 1, sentence_length, 1)
        2. representation vector for the sentence
            output size (batch_size, sentence_length)
        Parameters
        ----------
        x : 4-D torch Tensor
            convolution output of size (batch_size, 1, sentence_length, width)

        Returns
        -------
        output : 2-D torch Tensor
            representation vector of size (batch_size, width)
        '''
        return self.ap(x).squeeze(1).squeeze(2)


class WpLayer(nn.Module):
    '''column-wise averaging over windows of w consecutive columns
    Attributes
    ----------
    attention : bool
        compute layer with attention matrix
    '''

    def __init__(self, sentence_length, filter_width, attention):
        super(WpLayer, self).__init__()
        self.attention = attention
        if attention:
            self.sentence_length = sentence_length
            self.filter_width = filter_width
        else:
            self.wp = nn.AvgPool2d((filter_width, 1), stride=1)

    def forward(self, x, attention_matrix=None):
        '''
        if attention
            reweight the convolution output with attention matrix
        else
            average pooling
        Parameters
        ----------
        x : 4-D torch Tensor
            convolution output of size (batch_size, 1, sentence_length + filter_width - 1, height)
        attention_matrix: 2-D torch Tensor
            attention matrix between (convolution output x1 and convolution output x2) of size (batch_size, sentence_length + filter_width - 1)

        Returns
        -------
        output : 4-D torch Tensor
            size (batch_size, 1, sentence_length, height)
        '''
        if self.attention:
            pools = []
            attention_matrix = attention_matrix.unsqueeze(1).unsqueeze(3)
            for i in range(self.sentence_length):
                pools.append(
                    (x[:, :, i:i + self.filter_width, :] * attention_matrix[:, :, i:i + self.filter_width, :]).sum(
                        dim=2, keepdim=True))

            return torch.cat(pools, dim=2)

        else:
            return self.wp(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Layer') == -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
