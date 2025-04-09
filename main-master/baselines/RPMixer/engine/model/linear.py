# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear model
    """

    def __init__(self, seq_len, pred_len, seq_dim, feat_dim,
                 multidim_handle, is_normal):
        super(Linear, self).__init__()
        self.model_name = 'Linear'
        seq_feat_len = seq_len + feat_dim

        assert multidim_handle in ['individual', 'share', ]

        if multidim_handle == 'individual':
            encoder = {}
            decoder = {}
            for i in range(seq_dim):
                encoder[i] = nn.Identity()
                decoder[i] = nn.Linear(
                    seq_feat_len, pred_len)
                self.add_module(f'encoder_{i}', encoder[i])
                self.add_module(f'decoder_{i}', decoder[i])
        elif multidim_handle == 'share':
            encoder = nn.Identity()
            decoder = nn.Linear(
                seq_feat_len, pred_len)
            self.add_module('encoder', encoder)
            self.add_module('decoder', decoder)

        self.encoder = encoder
        self.decoder = decoder

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seq_dim = seq_dim
        self.multidim_handle = multidim_handle
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

        multidim_handle = self.multidim_handle
        if multidim_handle == 'individual':
            seq_dim = self.seq_dim
            y = []
            for i in range(seq_dim):
                h = self.encoder[i](x[:, i, :])
                y.append(self.decoder[i](h))
                y[i] = torch.unsqueeze(y[i], 1)
            y = torch.cat(y, dim=1)
        elif multidim_handle == 'share':
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

