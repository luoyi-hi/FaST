# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import joblib
import argparse
import numpy as np
from engine.util import check_dir
from engine.util import parse_config
from engine.data_io import get_ltsf_loader as get_loader
from engine.train_test import get_ltsf_metric as get_metric
from engine.model import OneNNRegr


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
    neighbor_dir = os.path.join(
        '.', 'neighbor', data_name)
    result_dir = os.path.join(
        '.', 'result', data_name)
    check_dir(neighbor_dir)
    check_dir(result_dir)

    neighbor_path = os.path.join(
        neighbor_dir, f'{method_name}.npz')
    result_path = os.path.join(
        result_dir, f'{method_name}.joblib')
    if os.path.isfile(result_path):
        return

    data_config = os.path.join(
        '.', 'config', f'{data_name}.config')
    data_config = parse_config(data_config, verbose=True)

    method_config = os.path.join(
        '.', 'config', f'{method_name}.config')
    method_config = parse_config(method_config, verbose=True)
    method_config['model']['model_name'] = '1NN'
    data_config['data']['seq_len'] = method_config['model']['seq_len']
    data_config['model'] = method_config['model']

    pred_len = int(data_config['data']['pred_len'])
    seq_len = int(method_config['model']['seq_len'])

    _, dataset_train = get_loader(
        'train', data_config, n_worker)
    _, dataset_valid = get_loader(
        'valid', data_config, n_worker)
    _, dataset_test = get_loader(
        'test', data_config, n_worker)

    model = OneNNRegr(
        seq_len, pred_len)
    model.fit(dataset_train.data)

    if os.path.isfile(neighbor_path):
        neighbor_pkl = np.load(neighbor_path)
        neighbor_train = neighbor_pkl['neighbor_train']
        neighbor_valid = neighbor_pkl['neighbor_valid']
        neighbor_test = neighbor_pkl['neighbor_test']

        neighbor_train = neighbor_train.astype(int)
        neighbor_valid = neighbor_valid.astype(int)
        neighbor_test = neighbor_test.astype(int)

        gtrue_valid, pred_valid = model.predict_from_neighbor(
            dataset_valid.data, neighbor_valid)
        gtrue_test, pred_test = model.predict_from_neighbor(
            dataset_test.data, neighbor_test)
    else:
        _, _, neighbor_train = model.predict(
            dataset_train.data, return_neighbor=True)
        gtrue_valid, pred_valid, neighbor_valid = model.predict(
            dataset_valid.data, return_neighbor=True)
        gtrue_test, pred_test, neighbor_test = model.predict(
            dataset_test.data, return_neighbor=True)
        np.savez_compressed(
            neighbor_path,
            neighbor_train=neighbor_train,
            neighbor_valid=neighbor_valid,
            neighbor_test=neighbor_test)

    metric_valid = get_metric(pred_valid, gtrue_valid)
    metric_test = get_metric(pred_test, gtrue_test)

    result_pkl = {}
    result_pkl['metric_valid'] = metric_valid
    result_pkl['metric_test'] = metric_test
    result_pkl['n_param'] = 0
    joblib.dump(result_pkl, result_path)


if __name__ == '__main__':
    main_wrapper()

