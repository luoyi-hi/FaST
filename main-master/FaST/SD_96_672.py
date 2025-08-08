import os
import sys
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../.."))

from basicts.metrics import masked_ae, masked_ape, masked_se
from basicts.data import MyTimeSeries
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings
import torch.nn as nn


def smooth_l1_loss_wrapper(prediction, target):
    loss_fn = nn.SmoothL1Loss(beta=1)
    return loss_fn(prediction, target)


from .arch import FaST

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'SD'
num_nodes = 716
INPUT_LEN = 96
OUTPUT_LEN = 672
NUM_EPOCHS = 50
BATCH_SIZE = 64

regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings[
    "TRAIN_VAL_TEST_RATIO"
]  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings[
    "NORM_EACH_CHANNEL"
]  # Whether to normalize each channel of the data
RESCALE = regular_settings["RESCALE"]  # Whether to rescale the data
NULL_VAL = regular_settings["NULL_VAL"]  # Null value in the data
MODEL_ARCH = FaST

# Model architecture and parameters
MODEL_PARAM = {
    "num_nodes": num_nodes,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "layers": 3,
    "num_experts": 8,
    "daily_steps": 96,
    "weekly_days": 7,
    "hidden_dim": 128,
    "num_agent": 32,
}

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = "An Example Config"
CFG["fp16"] = True
CFG.GPU_NUM = 1  # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

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

CFG.TRAIN.LOSS = smooth_l1_loss_wrapper
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [10, 20, 30, 40, 50], "gamma": 0.5}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PREFETCH = True 
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = True 

# Gradient clipping settings
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE
CFG.VAL.DATA.PREFETCH = True 
CFG.VAL.DATA.NUM_WORKERS = 4
CFG.VAL.DATA.PIN_MEMORY = True 

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 200
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TEST.DATA.PREFETCH = True 
CFG.TEST.DATA.NUM_WORKERS = 4 
CFG.TEST.DATA.PIN_MEMORY = True 

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = []  # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True  # Whether to use GPU for evaluation. Default: True