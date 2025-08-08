# -*- coding: utf-8 -*-
"""
@author:
"""


import numpy as np
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def _normalize(data):
    data_mu = np.mean(data)
    data_sigma = np.std(data)
    if data_sigma < 1e-06:
        data_sigma = 1
    data -= data_mu
    data /= data_sigma
    return data


def _convert_circular(data):
    data_0 = np.cos(data * 2 * np.pi)
    data_1 = np.sin(data * 2 * np.pi)
    return np.array(
        [
            data_0,
            data_1,
        ]
    )


class TimeSeries(Dataset):
    def __init__(
        self,
        data,
        tod,
        dow,
        lat,
        lng,
        direction,
        seq_len,
        pred_len,
        partition,
        frac_test=0.2,
        frac_valid=0.2,
        frac_train=0.6,
        feature_variant=1,
    ):
        assert partition in [
            "train",
            "valid",
            "test",
        ]

        self.seq_len = seq_len
        self.pred_len = pred_len

        import os

        sample_path = "/data/" + str(self.seq_len) + "_" + str(self.pred_len)
        train_idx = np.load(os.getcwd() + sample_path + "/idx_train.npy")
        val_idx = np.load(os.getcwd() + sample_path + "/idx_val.npy")
        test_idx = np.load(os.getcwd() + sample_path + "/idx_test.npy")

        n_data = data.shape[0]
        n_test = int(n_data * frac_test)
        n_valid = int(n_data * frac_valid)
        data_usage = frac_test + frac_valid + frac_train
        if np.isclose(data_usage, 1.0):
            n_train = n_data - n_test - n_valid
            n_nonuse = 0
        else:
            n_train = int(n_data * frac_train)
            n_nonuse = n_data - n_test - n_valid - n_train

        train_start = train_idx[0] - self.pred_len - self.seq_len + 1
        train_end = train_idx[-1] + 1
        data_mu = np.mean(data[train_start:train_end, :], axis=0, keepdims=True)
        data_sigma = np.std(data[train_start:train_end, :], axis=0, keepdims=True)
        data_sigma[0, data_sigma[0, :] < 1e-6] = 1

        self.data_mu = np.expand_dims(data_mu.T, 0)
        self.data_sigma = np.expand_dims(data_sigma.T, 0)

        if partition == "train":
            segment_start = train_idx[0] - self.pred_len - self.seq_len + 1
            segment_end = train_idx[-1] + 1
        elif partition == "valid":
            segment_start = val_idx[0] - self.pred_len - self.seq_len + 1
            segment_end = val_idx[-1] + 1
        elif partition == "test":
            segment_start = test_idx[0] - self.pred_len - self.seq_len + 1
            segment_end = test_idx[-1] + 1

        self.partition = partition
        self.feature_variant = feature_variant
        self.data = (data[segment_start:segment_end, :] - data_mu) / data_sigma
        self.tod = tod[segment_start:segment_end]
        self.dow = dow[segment_start:segment_end]

        self.lat = np.expand_dims(_normalize(lat), 1)
        self.lng = np.expand_dims(_normalize(lng), 1)
        self.direction = np.zeros((direction.shape[0], 4))
        self.direction[direction == 0, 0] = 1
        self.direction[direction == 1, 1] = 1
        self.direction[direction == 2, 2] = 1
        self.direction[direction == 3, 3] = 1
        self.direction[:, 0] = _normalize(self.direction[:, 0])
        self.direction[:, 1] = _normalize(self.direction[:, 1])
        self.direction[:, 2] = _normalize(self.direction[:, 2])
        self.direction[:, 3] = _normalize(self.direction[:, 3])
        self.n_data = self.data.shape[0] - seq_len - pred_len + 1

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        feature_variant = self.feature_variant

        seq_len = self.seq_len
        pred_len = self.pred_len

        x_start = index
        x_end = x_start + seq_len

        y_start = x_end
        y_end = y_start + pred_len

        data_x = self.data[x_start:x_end, :].T
        data_y = self.data[y_start:y_end, :].T

        if feature_variant == 0:
            pass  # no feature is used
        if feature_variant == 1:
            n_dim = data_x.shape[0]
            feat_x = np.concatenate(
                (
                    self.tod[x_start:x_end],
                    self.dow[x_start:x_end],
                ),
                axis=0,
            )
            feat_x = np.tile(feat_x, (n_dim, 1))
            data_x = np.concatenate(
                (
                    data_x,
                    feat_x,
                ),
                axis=1,
            )
        elif feature_variant == 2:
            n_dim = data_x.shape[0]
            tod_feat = _convert_circular(self.tod[x_end])
            dow_feat = _convert_circular(self.dow[x_end])

            time_feat = np.concatenate(
                (
                    tod_feat,
                    dow_feat,
                ),
                axis=0,
            )
            time_feat = np.tile(time_feat, (n_dim, 1))
            data_x = np.concatenate(
                (
                    data_x,
                    time_feat,
                    self.lat,
                    self.lng,
                    self.direction,
                ),
                axis=1,
            )
            # seq_len + 4 + 1 + 1 + 4
        return data_x, data_y

    def inverse_transform(self, data):
        data = (data * self.data_sigma) + self.data_mu
        return data


def get_time_series_loader(partition, config, n_worker):
    data_path = config["data"]["data_path"]

    seq_len = int(config["data"]["seq_len"])
    pred_len = int(config["data"]["pred_len"])
    frac_test = float(config["data"]["frac_test"])
    frac_valid = float(config["data"]["frac_valid"])
    frac_train = float(config["data"]["frac_train"])
    feature_variant = int(config["data"]["feature_variant"])

    batch_size = int(config["model"]["batch_size"])

    shuffle_flag = False
    drop_last_flag = False
    if partition == "train":
        shuffle_flag = True
        drop_last_flag = True

    data = np.load(data_path)["data"]
    tod = np.load(data_path)["tod"]
    dow = np.load(data_path)["dow"]
    lat = np.load(data_path)["lat"]
    lng = np.load(data_path)["lng"]
    direction = np.load(data_path)["direction"]
    dataset = TimeSeries(
        data,
        tod,
        dow,
        lat,
        lng,
        direction,
        seq_len,
        pred_len,
        partition,
        frac_test=frac_test,
        frac_valid=frac_valid,
        frac_train=frac_train,
        feature_variant=feature_variant,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=n_worker,
        drop_last=drop_last_flag,
    )
    return data_loader, dataset
