# -*- coding: utf-8 -*-
"""
@author: 
"""


import configparser
import pathlib
from collections import OrderedDict


def parse_config(config_path, verbose=False):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if verbose:
        print(config_path)
    config_dict = OrderedDict()
    for key_0 in parser:
        config_dict[key_0] = OrderedDict()
        for key_1 in parser[key_0]:
            val = parser[key_0][key_1]
            if val == 'None':
                val = None
            config_dict[key_0][key_1] = val
            if verbose:
                print(f'  {key_0}.{key_1}={val}')
    return config_dict


def check_dir(dir_path):
    path = pathlib.Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

