# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from engine.util import check_dir
from engine.config import get_datasets
from engine.config import get_methods


def main():
    ltsf_dir = '/path/to/the/ltsf_benchmark/output/folder/'
    largest_dir = '/path/to/the/largest/output/folder/'

    config_dir = os.path.join(
        '.', 'config')
    check_dir(config_dir)
    get_datasets(config_dir, largest_dir, ltsf_dir)
    get_methods(config_dir)


if __name__ == '__main__':
    main()

