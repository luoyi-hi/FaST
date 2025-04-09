# -*- coding: utf-8 -*-
"""
@author: 
"""


import time
from .linear import Linear
from .tsmixer import TSMixer
from .rpmixer import RPMixer


def get_rpmixer(config):
    seq_len = int(config['seq_len'])
    pred_len = int(config['pred_len'])
    seq_dim = int(config['seq_dim'])
    feat_dim = int(config['feat_dim'])
    proj_dim = int(config['proj_dim'])
    dim_factor = float(config['dim_factor'])
    n_layer = int(config['n_layer'])
    norm_layer = config['norm_layer']

    is_preact = config['is_preact']
    is_preact = is_preact.lower() == 'true'
    is_random = config['is_random']
    is_random = is_random.lower() == 'true'
    is_normal = config['is_normal']
    is_normal = is_normal.lower() == 'true'
    is_freq = config['is_freq']
    is_freq = is_freq.lower() == 'true'

    model = RPMixer(seq_len, pred_len, seq_dim, feat_dim,
                    proj_dim, dim_factor, n_layer, norm_layer,
                    is_preact, is_random, is_normal, is_freq)
    return model


def get_linear(config):
    seq_len = int(config['seq_len'])
    pred_len = int(config['pred_len'])
    seq_dim = int(config['seq_dim'])
    feat_dim = int(config['feat_dim'])
    multidim_handle = config['multidim_handle']

    is_normal = config['is_normal']
    is_normal = is_normal.lower() == 'true'

    model = Linear(seq_len, pred_len, seq_dim, feat_dim,
                   multidim_handle, is_normal)
    return model


def get_tsmixer(config):
    seq_len = int(config['seq_len'])
    pred_len = int(config['pred_len'])
    seq_dim = int(config['seq_dim'])
    feat_dim = int(config['feat_dim'])
    hidden_dim = int(config['hidden_dim'])
    n_block = int(config['n_block'])
    dropout = float(config['dropout'])

    is_normal = config['is_normal']
    is_normal = is_normal.lower() == 'true'

    model = TSMixer(seq_len, pred_len, seq_dim, feat_dim,
                    hidden_dim, n_block, dropout, is_normal)
    return model


def get_model(config, verbose=True):

    config = config['model']
    model_name = config['model_name']
    if verbose:
        print(f'get {model_name}... ', end='')
    tic = time.time()
    if model_name == 'Linear':
        model = get_linear(config)
    elif model_name == 'TSMixer':
        model = get_tsmixer(config)
    elif model_name == 'RPMixer':
        model = get_rpmixer(config)

    toc = time.time() - tic
    if verbose:
        print(f'done! {toc:0.2f}')
    return model

