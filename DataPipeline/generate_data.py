import pandas as pd
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

year = "2019"  # please specify the year, our experiments use 2019
ca_his = pd.read_hdf("ca_his_raw_" + year + ".h5")

### please comment this line if you don't want to do resampling
ca_his = ca_his.resample("15T").mean().round(0)

ca_his = ca_his.fillna(0)

ca_his.to_hdf("ca_his_" + year + ".h5", key="t", mode="w")

ca_meta = pd.read_csv("ca_meta.csv")
sd_meta = ca_meta[ca_meta.District == 11]
sd_meta = sd_meta.reset_index()
sd_meta = sd_meta.drop(columns=["index"])
sd_meta.to_csv("sd_meta.csv", index=False)

sd_meta_id2 = sd_meta.ID2.values.tolist()

ca_rn_adj = np.load("ca_rn_adj.npy")

sd_rn_adj = ca_rn_adj[sd_meta_id2]
sd_rn_adj = sd_rn_adj[:, sd_meta_id2]

np.save("sd_rn_adj.npy", sd_rn_adj)

sd_meta.ID = sd_meta.ID.astype(str)
sd_meta_id = sd_meta.ID.values.tolist()

sd_his = ca_his[sd_meta_id]
sd_his.to_hdf("sd_his_" + year + ".h5", key="t", mode="w")

ca_meta = pd.read_csv("ca_meta.csv")
gla_meta = ca_meta[
    (ca_meta.District == 7) | (ca_meta.District == 8) | (ca_meta.District == 12)
]
gla_meta = gla_meta.reset_index()
gla_meta = gla_meta.drop(columns=["index"])
gla_meta.to_csv("gla_meta.csv", index=False)

gla_meta_id2 = gla_meta.ID2.values.tolist()

gla_rn_adj = ca_rn_adj[gla_meta_id2]
gla_rn_adj = gla_rn_adj[:, gla_meta_id2]

np.save("gla_rn_adj.npy", gla_rn_adj)

gla_meta.ID = gla_meta.ID.astype(str)
gla_meta_id = gla_meta.ID.values.tolist()


gla_his = ca_his[gla_meta_id]

gla_his.to_hdf("gla_his_" + year + ".h5", key="t", mode="w")
