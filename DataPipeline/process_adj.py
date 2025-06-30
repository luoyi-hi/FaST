import numpy as np
import pickle
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_names = ["SD", "GBA", "GLA", "CA"]

for data_name in data_names:
    npy_file_path = f"{data_name}_rn_adj.npy"
    data = np.load(npy_file_path)

    pkl_file_path = f"../main-master/datasets/{data_name}/adj_mx.pkl"

    with open(pkl_file_path, "wb") as f:
        pickle.dump(data, f)
