# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch
import torch.nn as nn
from collections import OrderedDict


class _Block(nn.Module):
    """
    TSMixer block

    Modified from https://github.com/google-research/google-research/blob/master/tsmixer
    """

    def __init__(self, in_len, in_dim, hidden_dim, dropout):
        super(_Block, self).__init__()

        self.bn_0 = nn.BatchNorm2d(1)
        self.linear_0 = nn.Linear(in_len, in_len)
        self.relu_0 = nn.ReLU()
        self.dropout_0 = nn.Dropout(dropout)

        self.bn_1 = nn.BatchNorm2d(1)
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, in_dim)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, in_dim, x_len, ]
        # y = [batch_size, in_dim, x_len, ]

        h = torch.unsqueeze(x, 1)
        h = self.bn_0(h)
        h = h[:, 0, :, :]
        h = self.linear_0(h)
        h = self.relu_0(h)
        h = self.dropout_0(h)
        x = x + h

        h = torch.unsqueeze(x, 1)
        h = self.bn_1(h)
        h = h[:, 0, :, :]
        h = torch.transpose(h, 1, 2)
        h = self.linear_1(h)
        h = self.relu_1(h)
        h = self.dropout_1(h)
        h = self.linear_2(h)
        h = self.relu_2(h)
        h = self.dropout_2(h)
        h = torch.transpose(h, 1, 2)
        y = x + h
        return y


class TSMixer(nn.Module):
    """
    TSMixer

    Modified from https://github.com/google-research/google-research/blob/master/tsmixer
    """

    def __init__(self, seq_len, pred_len, seq_dim, feat_dim,
                 hidden_dim, n_block, dropout, is_normal):
        super(TSMixer, self).__init__()
        self.model_name = 'TSMixer'
        seq_feat_len = seq_len + feat_dim

        layers = OrderedDict()
        for i in range(n_block):
            layers[f'block_{i}'] = _Block(
                seq_feat_len, seq_dim, hidden_dim, dropout)
        encoder = nn.Sequential(layers)
        self.add_module('encoder', encoder)
        self.encoder = encoder
        self.decoder = nn.Linear(seq_feat_len, pred_len)

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.is_normal = is_normal

    def forward(self, x):
        # x = [batch_size, seq_dim, seq_len, ]
        # y = [batch_size, seq_dim, pred_len, ]
        is_normal = self.is_normal
        seq_len = self.seq_len
        if is_normal:
            x_ = x.detach()
            x_mu = torch.mean(x_[:, :, :seq_len], 2, keepdim=True)
            x_sigma = torch.std(x_[:, :, :seq_len], 2, keepdim=True)
            x_sigma[x_sigma < 1e-6] = 1.0
            x[:, :, :seq_len] = (x[:, :, :seq_len] - x_mu) / x_sigma

        h = self.encoder(x)
        y = self.decoder(h)

        if is_normal:
            y = (y * x_sigma) + x_mu
        return y

    def get_n_param(self):
        n_param = 0
        for param in self.parameters():
            n_param += torch.numel(param)
        return n_param

