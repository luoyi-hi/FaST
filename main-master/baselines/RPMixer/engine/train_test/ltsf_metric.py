# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np
import torch.nn as nn


def _MAE(pred, gtrue):
    return np.mean(np.abs(pred - gtrue))


def _MSE(pred, gtrue):
    return np.mean((pred - gtrue) ** 2)


def get_metric(pred, gtrue, is_mae_only=False):
    """
    Modified from https://github.com/cure-lab/LTSF-Linear
    """
    if is_mae_only:
        return _MAE(pred, gtrue)

    metric = {}
    metric['mae'] = _MAE(pred, gtrue)
    metric['mse'] = _MSE(pred, gtrue)
    return metric


def mse_loss(dataset, pred, gtrue, device):
    loss = nn.MSELoss()(pred, gtrue)
    return loss

