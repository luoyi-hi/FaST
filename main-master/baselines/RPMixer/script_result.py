# -*- coding: utf-8 -*-
"""
@author:
"""


import os
import joblib
import numpy as np
from scipy.stats import rankdata
from engine.util import check_dir


def print_largest_header():
    return (
        "config_name,n_param,mae,rmse,mape,mae,rmse,mape,"
        "mae,rmse,mape,mae,rmse,mape,"
    )


def print_largest_score(metric_test, method_name):
    config_score = (
        f"{metric_test['mae'][2]:.2f},"
        f"{metric_test['rmse'][2]:.2f},"
        f"{metric_test['mape'][2]:.2f},"
        f"{metric_test['mae'][5]:.2f},"
        f"{metric_test['rmse'][5]:.2f},"
        f"{metric_test['mape'][5]:.2f},"
        f"{metric_test['mae'][11]:.2f},"
        f"{metric_test['rmse'][11]:.2f},"
        f"{metric_test['mape'][11]:.2f},"
        f"{np.mean(metric_test['mae']):.4f},"
        f"{np.mean(metric_test['rmse']):.4f},"
        f"{np.mean(metric_test['mape']):.4f},"
    )
    sorting_score = [
        np.mean(metric_test["mae"]),
        np.mean(metric_test["rmse"]),
        np.mean(metric_test["mape"]),
    ]
    return config_score, sorting_score


def print_ltsf_header():
    return "config_name,n_param,mse,mae,"


def print_ltsf_score(metric_test, method_name):
    config_score = f"{metric_test['mse']:.3f}," f"{metric_test['mae']:.3f},"
    sorting_score = [
        metric_test["mse"],
        metric_test["mae"],
    ]
    return config_score, sorting_score


def print_table(method_names, data_name, print_score):
    output_str = ""
    config_strs = []
    config_scores = []
    sorting_scores = []

    n_total = 0
    n_done = 0

    result_dir = os.path.join(".", "result", f"{data_name}")
    for method_name in method_names:
        n_total += 1
        config_str = f"{method_name}"

        if "1nn" in method_name:
            result_path = os.path.join(result_dir, f"{method_name}.joblib")
        else:
            result_path = os.path.join(result_dir, f"{method_name}_9999.joblib")
        if not os.path.isfile(result_path):
            continue
        n_done += 1

        result_pkl = joblib.load(result_path)
        metric_test = result_pkl["metric_test"]
        config_score, sorting_score = print_score(metric_test, method_name)
        config_score = f"{result_pkl['n_param']}," + config_score

        config_strs.append(config_str)
        config_scores.append(config_score)
        sorting_scores.append(sorting_score)

    if n_done == 0:
        return ""
    sorting_scores = np.array(sorting_scores)
    sorting_scores = rankdata(sorting_scores, axis=0)
    sorting_scores = np.mean(sorting_scores, axis=1)
    order = np.argsort(sorting_scores)

    config_strs_ = []
    config_scores_ = []
    for idx in order:
        config_strs_.append(config_strs[idx])
        config_scores_.append(config_scores[idx])
    config_strs = config_strs_
    config_scores = config_scores_

    for config_str, config_score in zip(config_strs, config_scores):
        output_str += f"{config_str},{config_score}\n"
    return output_str


def main():
    result_dir = os.path.join(".", "result")
    check_dir(result_dir)
    result_path = os.path.join(result_dir, "result_table.csv")
    output_str = ""

    data_names = [
        "ca_his_2019_agg_48",
    ]
    method_names = [
        "rpm_96",
    ]

    for data_name in data_names:
        output_str += f"{data_name}\n"
        output_str += f"{print_largest_header()}\n"
        output_str += print_table(method_names, data_name, print_largest_score)
        output_str += f"\n"

    with open(result_path, "w") as f:
        f.write(output_str)


if __name__ == "__main__":
    main()
