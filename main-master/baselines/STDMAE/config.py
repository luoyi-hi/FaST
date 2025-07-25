import os
import sys
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_ae, masked_ape, masked_se, masked_mse_bts
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import MyZScoreScaler
from basicts.data import MyTimeSeries
from basicts.utils import get_regular_settings, load_adj

from .arch import STDMAE

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
MODEL_ARCH = STDMAE
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "dataset_name": DATA_NAME,
    "pre_trained_tmae_path": "checkpoints/Mask_TMAE/sd_100_96_48/Mask_TMAE_best_val_loss.pt",
    "pre_trained_smae_path": "checkpoints/Mask_SMAE/sd_100_96_48/Mask_SMAE_best_val_MAE.pt",
    "mask_args": {
        "patch_size": 12,
        "in_channel": 1,
        "embed_dim": 96,
        "num_heads": 4,
        "mlp_ratio": 4,
        "dropout": 0.1,
        "mask_ratio": 0.25,
        "encoder_depth": 4,
        "decoder_depth": 1,
        "mode": "forecasting",
    },
    "backend_args": {
        "num_nodes": num_nodes,
        "supports": [torch.tensor(i) for i in adj_mx],
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": 2,
        "out_dim": 48,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    },
    "short_term_len": INPUT_LEN,
}

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = "An Example Config"
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
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

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
    CFG.MODEL.NAME,
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 18, 36, 54, 72], "gamma": 0.5}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 1
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 3.0}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE
CFG.VAL.DATA.NUM_WORKERS = 1
CFG.VAL.DATA.PIN_MEMORY = True

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 200
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TEST.DATA.NUM_WORKERS = 1
CFG.TEST.DATA.PIN_MEMORY = True

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = []  # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True  # Whether to use GPU for evaluation. Default: True
