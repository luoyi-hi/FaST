# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import torch


def check_checkpoint(model_path, n_epoch):
    for i in range(n_epoch):
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            try:
                _ = torch.load(model_path_i, map_location='cpu')
            except:
                print(f'{model_path_i} can not be opened. It is removed!')
                os.remove(model_path_i)

