# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from .util import write_config


def get_methods(config_dir):
    prefix = '1nn'
    config_id = 0
    config_path = os.path.join(
        config_dir, f'{prefix}_{config_id:04d}.config')
    config_dict = _get_1nneighbor()
    write_config(config_dict, config_path)

    prefix = 'lin'
    config_id = 0
    config_path = os.path.join(
        config_dir, f'{prefix}_{config_id:04d}.config')
    config_dict = _get_linear()
    write_config(config_dict, config_path)

    prefix = 'tsm'
    config_id = 0
    config_path = os.path.join(
        config_dir, f'{prefix}_{config_id:04d}.config')
    config_dict = _get_tsmixer()
    write_config(config_dict, config_path)

    prefix = 'rpm'
    config_id = 0
    for config_dict in _get_rpmixer():
        config_path = os.path.join(
            config_dir, f'{prefix}_{config_id:04d}.config')
        write_config(config_dict, config_path)
        config_id += 1


def _get_1nneighbor():
    seq_len = 96
    batch_size = 32

    config = {}
    config['model'] = {}
    config['model']['model_name'] = '1NN'
    config['model']['seq_len'] = seq_len
    config['model']['batch_size'] = batch_size
    return config


def _get_linear():
    seq_len = 96
    multidim_handle = 'share'
    is_normal = 'True'

    lr = 0.001
    n_epoch = 100
    batch_size = 32

    config = {}
    config['model'] = {}
    config['model']['model_name'] = 'Linear'
    config['model']['seq_len'] = seq_len
    config['model']['multidim_handle'] = multidim_handle
    config['model']['is_normal'] = is_normal

    config['model']['lr'] = lr
    config['model']['n_epoch'] = n_epoch
    config['model']['batch_size'] = batch_size
    return config


def _get_tsmixer():
    seq_len = 96
    hidden_dim = 64
    n_block = 8
    dropout = 0.0
    is_normal = 'True'

    lr = 0.001
    n_epoch = 100
    batch_size = 32

    config = {}
    config['model'] = {}
    config['model']['model_name'] = 'TSMixer'
    config['model']['seq_len'] = seq_len
    config['model']['hidden_dim'] = hidden_dim
    config['model']['n_block'] = n_block
    config['model']['dropout'] = dropout
    config['model']['is_normal'] = is_normal

    config['model']['lr'] = lr
    config['model']['n_epoch'] = n_epoch
    config['model']['batch_size'] = batch_size
    return config


def _get_rpmixer_default():
    seq_len = 96
    proj_dim = -1
    dim_factor = 1.0
    n_layer = 8
    norm_layer = 'None'
    is_preact = 'True'
    is_random = 'True'
    is_normal = 'True'
    is_freq = 'True'

    lr = 0.001
    n_epoch = 100
    batch_size = 32

    config = {}
    config['model'] = {}
    config['model']['model_name'] = 'RPMixer'
    config['model']['seq_len'] = seq_len
    config['model']['proj_dim'] = proj_dim
    config['model']['dim_factor'] = dim_factor
    config['model']['n_layer'] = n_layer
    config['model']['norm_layer'] = norm_layer
    config['model']['is_preact'] = is_preact
    config['model']['is_random'] = is_random
    config['model']['is_normal'] = is_normal
    config['model']['is_freq'] = is_freq

    config['model']['lr'] = lr
    config['model']['n_epoch'] = n_epoch
    config['model']['batch_size'] = batch_size
    return config


def _get_rpmixer():
    config = _get_rpmixer_default()
    yield config

    config = _get_rpmixer_default()
    config['model']['seq_len'] = 512
    config['model']['n_layer'] = 1
    config['model']['is_normal'] = 'False'
    config['model']['lr'] = 0.0001
    yield config

    config = _get_rpmixer_default()
    config['model']['seq_len'] = 512
    config['model']['n_layer'] = 1
    config['model']['lr'] = 0.0001
    yield config

    config = _get_rpmixer_default()
    config['model']['seq_len'] = 512
    config['model']['proj_dim'] = 64
    config['model']['n_layer'] = 2
    config['model']['lr'] = 0.0001
    yield config

    config = _get_rpmixer_default()
    config['model']['seq_len'] = 512
    config['model']['n_layer'] = 4
    config['model']['lr'] = 0.0001
    yield config

    config = _get_rpmixer_default()
    config['model']['seq_len'] = 512
    config['model']['proj_dim'] = 32
    config['model']['n_layer'] = 1
    config['model']['lr'] = 0.0001
    yield config

