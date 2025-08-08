# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np
import pyscamp


class OneNNRegr:
    def __init__(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len

    def fit(self, data_train):
        self.data_train = data_train

    def predict_from_neighbor(self, data, neighbor):
        seq_len = self.seq_len
        pred_len = self.pred_len
        n_sub = neighbor.shape[0]
        n_dim = neighbor.shape[1]

        gtrue = np.zeros((n_sub, n_dim, pred_len, ))
        for i in range(n_sub):
            gtrue[i, :, :] = data[i + seq_len:i + seq_len + pred_len, :].T

        pred = np.zeros((n_sub, n_dim, pred_len, ))
        for i in range(n_dim):
            neighbor_idx = neighbor[:, i]

            pred[:, i, :] = self._scale_predict(
                data[:, i], self.data_train[:, i],
                neighbor_idx)
        return gtrue, pred

    def predict(self, data, return_neighbor=False):
        seq_len = self.seq_len
        pred_len = self.pred_len

        n_sub = data.shape[0] - seq_len - pred_len + 1
        n_dim = data.shape[1]
        assert n_dim == self.data_train.shape[1]

        gtrue = np.zeros((n_sub, n_dim, pred_len, ))
        for i in range(n_sub):
            gtrue[i, :, :] = data[i + seq_len:i + seq_len + pred_len, :].T

        pred = np.zeros((n_sub, n_dim, pred_len, ))
        if return_neighbor:
            neighbor = np.zeros((n_sub, n_dim, ))

        for i in range(n_dim):
            if (data.shape[0] == self.data_train.shape[0] and
                    np.allclose(data[:, i], self.data_train[:, i])):
                _, neighbor_idx = pyscamp.selfjoin(
                    data[:-pred_len, i],
                    seq_len)
            else:
                _, neighbor_idx = pyscamp.abjoin(
                    data[:-pred_len, i],
                    self.data_train[:-pred_len, i],
                    seq_len)
            if return_neighbor:
                neighbor[:, i] = neighbor_idx

            pred[:, i, :] = self._scale_predict(
                data[:, i], self.data_train[:, i],
                neighbor_idx)

        if return_neighbor:
            return gtrue, pred, neighbor
        else:
            return gtrue, pred

    def _scale_predict(self, data_test, data_train, neighbor_idx):
        seq_len = self.seq_len
        pred_len = self.pred_len

        n_sub = data_test.shape[0] - seq_len - pred_len + 1

        pred = np.zeros((n_sub, pred_len, ))
        for i in range(n_sub):
            query_i = data_test[i:i + seq_len]
            query_i_max = np.max(query_i)
            query_i_min = np.min(query_i)
            query_i_range = query_i_max - query_i_min
            if query_i_range < 1e-6:
                query_i_range = 1
            query_i_norm = ((query_i - query_i_min) /
                            query_i_range)

            neighbor_i_idx = neighbor_idx[i]
            if neighbor_i_idx == -1:
                pred[i, :] = np.ones(pred_len) * query_i[-1]
            else:
                neighbor_i = data_train[
                    neighbor_i_idx:
                    neighbor_i_idx + seq_len]
                neighbor_i_max = np.max(neighbor_i)
                neighbor_i_min = np.min(neighbor_i)
                neighbor_i_range = neighbor_i_max - neighbor_i_min
                if neighbor_i_range < 1e-6:
                    neighbor_i_range = 1
                neighbor_i_norm = ((neighbor_i - neighbor_i_min) /
                                   neighbor_i_range)

                pred_i = data_train[
                    neighbor_i_idx + seq_len:
                    neighbor_i_idx + seq_len + pred_len]
                pred_i_norm = ((pred_i - neighbor_i_min) /
                               neighbor_i_range)
                offset = - neighbor_i_norm[-1] + query_i_norm[-1]
                pred[i, :] = ((pred_i_norm + offset) *
                              query_i_range + query_i_min)
        return pred

