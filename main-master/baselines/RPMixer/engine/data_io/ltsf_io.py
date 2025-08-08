# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TimeSeries(Dataset):
    def __init__(self, data_name, data, seq_len, pred_len, partition):
        assert partition in ['train', 'valid', 'test', ]

        self.seq_len = seq_len
        self.pred_len = pred_len

        n_data = data.shape[0]
        if data_name in ['ETTm1', 'ETTm1', ]:
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif data_name in ['ETTh1', 'ETTh1', ]:
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n_data * 0.7)
            val_end = n_data - int(n_data * 0.2)
            test_end = n_data

        data_mu = np.mean(data[:train_end, :],
                          axis=0, keepdims=True)
        data_sigma = np.std(data[:train_end, :],
                            axis=0, keepdims=True)
        data_sigma[0, data_sigma[0, :] < 1e-6] = 1

        self.data_mu = np.expand_dims(data_mu.T, 0)
        self.data_sigma = np.expand_dims(data_sigma.T, 0)

        if partition == 'train':
            segment_start = 0
            segment_end = train_end
        elif partition == 'valid':
            segment_start = train_end - seq_len
            segment_end = val_end
        elif partition == 'test':
            segment_start = val_end - seq_len
            segment_end = n_data

        self.data = ((data[segment_start:segment_end, :] - data_mu) /
                     data_sigma)
        self.n_data = self.data.shape[0] - seq_len - pred_len + 1

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        seq_len = self.seq_len
        pred_len = self.pred_len

        x_start = index
        x_end = x_start + seq_len

        y_start = x_end
        y_end = y_start + pred_len

        data_x = self.data[x_start:x_end, :].T
        data_y = self.data[y_start:y_end, :].T
        return data_x, data_y

    def inverse_transform(self, data):
        data = (data * self.data_sigma) + self.data_mu
        return data


def get_time_series_loader(partition, config, n_worker):
    data_path = config['data']['data_path']

    data_name = config['data']['data_name']
    seq_len = int(config['data']['seq_len'])
    pred_len = int(config['data']['pred_len'])
    batch_size = int(config['model']['batch_size'])

    shuffle_flag = False
    drop_last_flag = False
    if partition == 'train':
        shuffle_flag = True
        drop_last_flag = True

    data = np.load(data_path)['data']

    dataset = TimeSeries(
        data_name, data, seq_len, pred_len, partition)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=n_worker,
        drop_last=drop_last_flag)
    return data_loader, dataset

