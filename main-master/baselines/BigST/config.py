import os
import sys
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_ae, masked_ape, masked_se, masked_mse_bts
from basicts.data import MyTimeSeries
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import MyZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import BigST

# from .runner import BigSTPreprocessRunner
from .loss import bigst_loss

import pdb

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'sd'
num_nodes = 716
INPUT_LEN = 96
OUTPUT_LEN = 48
NUM_EPOCHS = 50
BATCH_SIZE = 32


regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings[
    "TRAIN_VAL_TEST_RATIO"
]  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings[
    "NORM_EACH_CHANNEL"
]  # Whether to normalize each channel of the data
RESCALE = regular_settings["RESCALE"]  # Whether to rescale the data
NULL_VAL = regular_settings["NULL_VAL"]  # Null value in the data
# Model architecture and parameters
PREPROCESSED_FILE = (
    "checkpoints/BigSTPreprocess/sd_100_672_48/BigSTPreprocess_best_val_MAE.pt"
)
MODEL_ARCH = BigST

adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "bigst_args": {
        "num_nodes": num_nodes,
        "seq_num": INPUT_LEN,
        "in_dim": 3,
        "out_dim": OUTPUT_LEN,  #  源代码固定成12了
        "hid_dim": 32,
        "tau": 0.25,
        "random_feature_dim": 64,
        "node_emb_dim": 32,
        "time_emb_dim": 32,
        "use_residual": True,
        "use_bn": True,
        "use_long": False,
        "use_spatial": True,
        "dropout": 0.3,
        "supports": [torch.tensor(i) for i in adj_mx],
        "time_of_day_size": 96,
        "day_of_week_size": 7,
    },
    "preprocess_path": PREPROCESSED_FILE,
    "preprocess_args": {
        "num_nodes": num_nodes,
        "in_dim": 3,
        "dropout": 0.3,
        "input_length": INPUT_LEN,
        "output_length": OUTPUT_LEN,
        "nhid": 32,
        "tiny_batch_size": 700,
    },
}

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = "An Example Config"
CFG.GPU_NUM = 1  # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Environment Configuration ##############################

CFG.ENV = EasyDict()  # Environment settings. Default: None
CFG.ENV.SEED = 0  # Random seed. Default: None

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = MyTimeSeries
CFG.DATASET.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        # 'mode' is automatically set by the runner
    }
)

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = MyZScoreScaler  # Scaler class
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": NORM_EACH_CHANNEL,
        "rescale": RESCALE,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
)

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict(
    {
        "MAE": masked_ae,
        "RMSE": masked_se,
        "MAPE": masked_ape,
    }
)
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
)


CFG.TRAIN.LOSS = bigst_loss if MODEL_PARAM["bigst_args"]["use_spatial"] else masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 50], "gamma": 0.5}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PREFETCH = True # 是否使用预取的数据加载器。详见 https://github.com/justheuristic/prefetch_generator。默认值：False。
CFG.TRAIN.DATA.NUM_WORKERS = 4 # 训练数据加载器的工作线程数。默认值：0
CFG.TRAIN.DATA.PIN_MEMORY = True # 训练数据加载器是否固定内存。默认值：False

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 200
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.USE_GPU = True  # Whether to use GPU for evaluation. Default: True
