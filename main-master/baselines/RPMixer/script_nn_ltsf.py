# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import copy
import argparse
import torch
from engine.util import check_dir
from engine.util import parse_config
from engine.data_io import get_seq_dim
from engine.data_io import get_ltsf_loader as get_loader
from engine.model import get_model
from engine.train_test import nn_train
from engine.train_test import ltsf_mse_loss as loss_fun
from engine.train_test import get_ltsf_metric as get_metric


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name')
    parser.add_argument('--method_name')
    parser.add_argument('--n_worker', type=int)

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name
    n_worker = args.n_worker
    main(data_name, method_name, n_worker)


def main(data_name, method_name, n_worker):
    model_dir = os.path.join(
        '.', 'model', data_name)
    result_dir = os.path.join(
        '.', 'result', data_name)
    check_dir(model_dir)
    check_dir(result_dir)

    data_config = os.path.join(
        '.', 'config', f'{data_name}.config')
    data_config = parse_config(data_config, verbose=True)

    method_config = os.path.join(
        '.', 'config', f'{method_name}.config')
    method_config = parse_config(method_config, verbose=True)

    config = copy.deepcopy(method_config)
    config['data'] = data_config['data']
    config['data']['seq_len'] = config['model']['seq_len']
    config['model']['pred_len'] = config['data']['pred_len']
    config['model']['seq_dim'] = get_seq_dim(config)
    config['model']['feat_dim'] = 0

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)

    fmt_str = '{0:04d}'
    model_path = os.path.join(
        model_dir, f'{method_name}_{fmt_str}.pt')
    result_path = os.path.join(
        result_dir, f'{method_name}_{fmt_str}.joblib')

    is_inverse = False
    model = get_model(config)
    nn_train(model, model_path, result_path, config,
             get_loader, loss_fun, get_metric, is_inverse,
             device, n_worker)


if __name__ == '__main__':
    main_wrapper()

