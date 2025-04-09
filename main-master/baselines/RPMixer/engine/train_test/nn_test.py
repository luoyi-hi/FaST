# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np
import torch


def nn_test(data_loader, dataset, model, get_metric, device,
            is_inverse, is_mae_only):
    pred = []
    gtrue = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)
            pred_i = model.forward(batch_x)

            pred_i = pred_i.detach().cpu().numpy()
            gtrue_i = batch_y.detach().cpu().numpy()
            pred.append(pred_i)
            gtrue.append(gtrue_i)

    pred = np.concatenate(pred, axis=0)
    gtrue = np.concatenate(gtrue, axis=0)

    if is_inverse:
        pred = dataset.inverse_transform(pred)
        gtrue = dataset.inverse_transform(gtrue)
    if is_mae_only:
        mae = get_metric(pred, gtrue, is_mae_only=True)
        return np.mean(mae)
    else:
        return get_metric(pred, gtrue)

