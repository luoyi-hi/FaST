# -*- coding: utf-8 -*-
"""
@author:
"""


import argparse
import random
from script_nn_largest import main as main_nn_largest
from script_nn_ltsf import main as main_nn_ltsf
from script_1nn_largest import main as main_1nn_largest
import torch
import numpy as np


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_worker", type=int, default=32)

    args = parser.parse_args()
    n_worker = args.n_worker

    exp_setup = []

    data_names = [
        "gla_his_2019_agg_48",
    ]

    method_names = [
        "rpm_96",
    ]
    for data_name in data_names:
        for method_name in method_names:
            exp_setup.append(
                [
                    data_name,
                    method_name,
                    main_nn_largest,
                ]
            )

    random.shuffle(exp_setup)
    for data_name, method_name, main_fun in exp_setup:
        print(data_name, method_name)
        main_fun(data_name, method_name, n_worker)


if __name__ == "__main__":
    set_random_seed()
    main()
