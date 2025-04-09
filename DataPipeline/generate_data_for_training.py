import os
import argparse
import numpy as np
import pandas as pd


def generate_data(df, add_time_of_day, add_day_of_week):
    _, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]

    if add_time_of_day:
        idx = df.index.values
        time_ind = (idx % 96) / 96.0
        time_ind = time_ind.reshape((-1, 1, 1))
        time_of_day = np.tile(time_ind, (1, num_nodes, 1))
        feature_list.append(time_of_day)

    if add_day_of_week:
        idx = df.index.values
        dow = ((idx // 96) % 7) / 7.0
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)

    return data


def generate_train_val_test(args):
    years = args.years.split("_")
    df = pd.DataFrame()
    for y in years:
        df_tmp = pd.read_hdf(args.dataset + "_his_" + y + ".h5")
        df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

    data = generate_data(df, args.tod, args.dow)

    # save
    out_dir = "../main-master/datasets/" + args.dataset + "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, "his.npz"), data=data)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ca", help="dataset name")
    parser.add_argument(
        "--years",
        type=str,
        default="2019",
        help="if use data from multiple years, please use underline to separate them, e.g., 2018_2019",
    )
    parser.add_argument("--tod", type=int, default=1, help="time of day")
    parser.add_argument("--dow", type=int, default=1, help="day of week")

    args = parser.parse_args()
    generate_train_val_test(args)
