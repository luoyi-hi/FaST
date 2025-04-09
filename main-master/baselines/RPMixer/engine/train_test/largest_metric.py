# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch
import numpy as np


def _MAE(pred, gtrue):
    mask = np.round(gtrue).astype(int) > 0
    mask = mask.astype(float)
    mask /= np.mean(mask)
    loss = np.abs(pred - gtrue)
    loss = loss * mask
    loss[np.isnan(loss)] = 0
    return np.mean(loss)


def _MSE(pred, gtrue):
    mask = np.round(gtrue).astype(int) > 0
    mask = mask.astype(float)
    mask /= np.mean(mask)
    loss = (pred - gtrue) ** 2
    loss = loss * mask
    loss[np.isnan(loss)] = 0
    return np.mean(loss)


def _RMSE(pred, gtrue):
    return np.sqrt(_MSE(pred, gtrue))


def _MAPE(pred, gtrue):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.round(gtrue).astype(int) > 0
        mask = mask.astype('float')
        mask /= mask.mean()
        mape = np.abs((pred - gtrue) / gtrue)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def get_metric(pred, gtrue, is_mae_only=False):
    pred_len = pred.shape[2]
    mae = np.zeros(pred_len)
    if not is_mae_only:
        rmse = np.zeros(pred_len)
        mape = np.zeros(pred_len)
    for i in range(pred_len):
        mae[i] = _MAE(pred[:, :, i], gtrue[:, :, i])
        if not is_mae_only:
            rmse[i] = _RMSE(pred[:, :, i], gtrue[:, :, i])
            mape[i] = _MAPE(pred[:, :, i], gtrue[:, :, i])

    if is_mae_only:
        return mae
    else:
        metric = {}
        metric['mae'] = mae
        metric['rmse'] = rmse
        metric['mape'] = mape
        return metric


def mae_loss(dataset, pred, gtrue, device):
    gtrue_np = gtrue.detach().cpu().numpy()
    gtrue_np = dataset.inverse_transform(gtrue_np)
    mask = np.round(gtrue_np).astype(int) > 0
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = torch.from_numpy(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask = mask.to(device, dtype=torch.float32)

    loss = torch.abs(pred - gtrue)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

