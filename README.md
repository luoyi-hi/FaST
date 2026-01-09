<div align="center">

# FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graphs via Mixture-of-Experts

[![preprint](https://img.shields.io/static/v1?label=FaST-Paper&message=2403.07815&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2601.05174v1)

</div>

## 1. Experimental Details

### 1.1 Experimental Setting


The experimental evaluation is implemented using the `BasicTS` framework. The maximum number of training epochs for all methods is set to 50, with early stopping based on validation set performance to select the optimal model parameters. Performance is evaluated using MAE, RMSE, and MAPE metrics. All experiments are conducted on a system equipped with an AMD EPYC 7532 processor at 2.40 GHz, an NVIDIA RTX A6000 GPU with 48 GB of memory, 128 GB of RAM, and Ubuntu 20.04. The default deep learning library is PyTorch version 2.2.1, with Python version 3.11.8.

The FaST model employs the Adam optimizer with an initial learning rate of 0.002 and a weight decay parameter of 0.0001 for regularization. Mixed precision training is utilized to enhance computational efficiency and reduce memory usage. During training, the learning rate scheduling strategy utilizes MultiStepLR, which decays the learning rate by a factor of 0.5 every 10 epochs, starting from the 10th epoch, to facilitate multi-stage progressive optimization and promote stable model convergence.



### 1.2 Dataset Description

The CA dataset used in our report was collected from the Performance Measurement System (PeMS) by the authors of [1], and we obtained the data through that work. The San Diego (SD), Greater Los Angeles (GLA), and Greater Bay Area (GBA) areas are three representative subregions selected from the CA dataset, containing 716, 3834, and 2352 sensors, respectively. 


The dataset can be downloaded from the following link: https://www.kaggle.com/datasets/liuxu77/largest. The link contains seven files. To reproduce our experiment results, you need to download the following three files: `ca_his_raw_2019.h5`, `ca_meta.csv`, `ca_rn_adj.npy`.


Install environment dependencies using the following command:

```shell
pip install -r requirements.txt
```


Unzip the downloaded data into the `DataPipeline` directory. Then, use the following command to generate the traffic data required for model training:


```shell
bash DataPipeline.sh
```

### 1.3 Data Generation for Model Training


We use the 2019 SD, GBA, GLA, and CA datasets. First, we obtain all samples through a sliding window, then split the samples into training, validation, and test sets in a 6:2:2 ratio.
The generated data will be stored in the `main-master/datasets` directory. In each data directory, the `his.npz` file stores raw traffic flow values along with derived daily and weekly features. The `adj_mx.pkl` file contains the adjacency matrix for the data, and `desc.json` stores the data information. Other folders, such as `{input_len}_{output_len}`, store the sample indices for the training, validation, and test sets for the corresponding forecasting length.


### 1.4 Training FaST Model


Run the following commands to train the FaST on different datasets and forecasting lengths:

```shell
# FaST on SD dataset
python main-master/experiments/train_seed.py -c FaST/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c FaST/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c FaST/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c FaST/SD_96_672.py -g 0

# FaST on GBA dataset
python main-master/experiments/train_seed.py -c FaST/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c FaST/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c FaST/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c FaST/GBA_96_672.py -g 0

# FaST on GLA dataset
python main-master/experiments/train_seed.py -c FaST/GLA_96_48.py -g 0
python main-master/experiments/train_seed.py -c FaST/GLA_96_96.py -g 0
python main-master/experiments/train_seed.py -c FaST/GLA_96_192.py -g 0
python main-master/experiments/train_seed.py -c FaST/GLA_96_672.py -g 0

# FaST on CA dataset
python main-master/experiments/train_seed.py -c FaST/CA_96_48.py -g 0
python main-master/experiments/train_seed.py -c FaST/CA_96_96.py -g 0
python main-master/experiments/train_seed.py -c FaST/CA_96_192.py -g 0
python main-master/experiments/train_seed.py -c FaST/CA_96_672.py -g 0
```


### 1.5 Baseline Reproduction

Use the following commands to reproduce baseline models:

```shell
# STID
bash script/STID.sh

# DLinear
bash script/DLinear.sh

# NHITS
bash script/NHITS.sh

# CycleNet
bash script/CycleNet.sh

# DCRNN
bash script/DCRNN.sh

# BigST
bash script/BigST.sh

# STGCN
bash script/STGCN.sh

# STPGNN
bash script/STPGNN.sh

# GWNet
bash script/GWNet.sh

# STDMAE
# Please add the paths of the two pre-trained models to the configuration file of STDMAE.
bash script/STDMAE.sh

# PatchSTG
bash script/PatchSTG.sh

# SGP
# Please refer to: ‘https://github.com/Graph-Machine-Learning-Group/sgp’ to configure the relevant environment
bash script/SGP.sh

# RPMixer
# Please refer to: ‘https://sites.google.com/view/rpmixer’ to configure the relevant environment
bash script/RPMixer.sh

```



