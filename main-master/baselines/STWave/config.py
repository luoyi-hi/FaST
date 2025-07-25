import os
import sys
import numpy as np
import scipy.sparse as sp
from easydict import EasyDict
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_ae, masked_ape, masked_se, masked_mse_bts
from basicts.data import MyTimeSeries
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import MyZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import STWave
from .loss import stwave_masked_mae


def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def loadGraph(adj_mx, hs, ls):
    graphwave = get_eigv(adj_mx+np.eye(adj_mx.shape[0]), hs)
    sampled_nodes_number = int(np.around(math.log(adj_mx.shape[0]))+2)*ls
    graph = csr_matrix(adj_mx)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    adj_gat = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
    return adj_gat, graphwave

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'sd'
num_nodes = 716
INPUT_LEN = 96
OUTPUT_LEN = 48
NUM_EPOCHS = 50
BATCH_SIZE = 32


regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
adj_mx, _ = load_adj("datasets/" + DATA_NAME +
                     "/adj_mx.pkl", "original")
adjgat, gwv = loadGraph(_, 128, 1)
MODEL_ARCH = STWave
MODEL_PARAM = {
    "input_dim": 1,
    "hidden_size": 128,
    "layers": 2,
    "seq_len": INPUT_LEN,
    "horizon": OUTPUT_LEN,
    "log_samples": 1,
    "adj_gat": adjgat,
    "graphwave": gwv,
    "time_in_day_size": 96,
    "day_in_week_size": 7,
    "wave_type": "coif1",
    "wave_levels": 2,
}

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = MyTimeSeries
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = MyZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

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
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = stwave_masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [100, 105],
    "gamma": 0.1
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
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
