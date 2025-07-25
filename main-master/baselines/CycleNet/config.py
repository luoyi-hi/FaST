import os
import sys
import math
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_ae, masked_ape, masked_se, masked_mse_bts
from basicts.data import MyTimeSeries
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import MyZScoreScaler
from basicts.utils import get_regular_settings, load_dataset_desc

from .arch import CycleNet

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
MODEL_ARCH = CycleNet
MODEL_PARAM = {
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "enc_in": num_nodes,
    "cycle_pattern": "daily&weekly",  # daily OR daily&weekly
    "cycle": 96,  # time_of_day_size
    "model_type": "mlp",  # linear or mlp
    "d_model": 512,
    "use_revin": True,
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
CFG.ENV.SEED = 42  # Random seed. Default: None

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
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.01}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
# CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
# CFG.TRAIN.LR_SCHEDULER.PARAM = {
#     "milestones": [1, 25, 50],
#     "gamma": 0.5
# }
desc = load_dataset_desc(DATA_NAME)
train_steps = math.ceil(desc["num_time_steps"] * TRAIN_VAL_TEST_RATIO[0])
CFG.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "pct_start": 0.3,
    "epochs": NUM_EPOCHS,
    "steps_per_epoch": train_steps,
    "max_lr": CFG.TRAIN.OPTIM.PARAM["lr"],
}
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
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