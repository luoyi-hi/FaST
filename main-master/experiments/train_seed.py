# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import basicts
import numpy as np

torch.set_num_threads(4)  # aviod high cpu avg usage

import random


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = ArgumentParser(
        description="Run time series forecasting model in BasicTS framework!"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        default="main-master/FaST/sd_96_48.py",
        help="training config",
    )
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    return parser.parse_args()


def main():
    set_random_seed()
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == "__main__":
    main()
