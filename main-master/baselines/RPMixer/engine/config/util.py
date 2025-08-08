# -*- coding: utf-8 -*-
"""
@author: 
"""


import configparser


def write_config(config_dict, config_path):
    config_parser = configparser.ConfigParser()
    for key in config_dict:
        config_parser[key] = config_dict[key]

    with open(config_path, 'w') as f:
        config_parser.write(f)
