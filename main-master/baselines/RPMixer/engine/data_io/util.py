# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np


def get_seq_dim(config):
    data_path = config['data']['data_path']
    data = np.load(data_path)['data']
    return data.shape[1]

