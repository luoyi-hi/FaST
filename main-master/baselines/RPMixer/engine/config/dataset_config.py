# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from .util import write_config


def _get_largest_dataset(config_dir, data_dir, data_name, pred_len, config_id):
    data_path = os.path.join(data_dir, f'{data_name}.npz')

    frac_test = 0.2
    frac_valid = 0.2
    frac_train = 0.6

    feature_variant = 1

    config_dict = {}
    config_dict['data'] = {}
    config_dict['data']['data_name'] = data_name
    config_dict['data']['data_path'] = data_path
    config_dict['data']['pred_len'] = pred_len
    config_dict['data']['frac_test'] = frac_test
    config_dict['data']['frac_valid'] = frac_valid
    config_dict['data']['frac_train'] = frac_train
    config_dict['data']['feature_variant'] = feature_variant

    config_path = os.path.join(
        config_dir, f'{data_name}_{config_id:04d}.config')
    write_config(config_dict, config_path)


def _get_ltsf_dataset(config_dir, data_dir, data_name, pred_len, config_id):
    data_path = os.path.join(data_dir, f'{data_name}.npz')

    config_dict = {}
    config_dict['data'] = {}
    config_dict['data']['data_name'] = data_name
    config_dict['data']['data_path'] = data_path
    config_dict['data']['pred_len'] = pred_len

    config_path = os.path.join(
        config_dir, f'{data_name}_{config_id:04d}.config')
    write_config(config_dict, config_path)



def get_datasets(config_dir, largest_dir, ltsf_dir):
    pred_len = 12
    data_names = [
        'ca_his_2019_agg',
        'gba_his_2019_agg',
        'gla_his_2019_agg',
        'sd_his_2019_agg',
    ]

    for data_name in data_names:
        config_id = 0
        _get_largest_dataset(
            config_dir, largest_dir, data_name, pred_len, config_id)


    pred_lens = [96, 192, 336, 720, ]
    data_names = [
        'Electricity',
        'Traffic',
        'Weather',
        'ETTh1',
        'ETTh2',
        'ETTm1',
        'ETTm2',
    ]
    for data_name in data_names:
        config_id = 0
        for pred_len in pred_lens:
            _get_ltsf_dataset(
                config_dir, ltsf_dir, data_name, pred_len, config_id)
            config_id += 1

