# DLinear on SD dataset
python main-master/experiments/train_seed.py -c baselines/DLinear/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/SD_96_672.py -g 0

# DLinear on GBA dataset
python main-master/experiments/train_seed.py -c baselines/DLinear/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/GBA_96_672.py -g 0

# DLinear on GLA dataset
python main-master/experiments/train_seed.py -c baselines/DLinear/GLA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/GLA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/GLA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/GLA_96_672.py -g 0

# DLinear on CA dataset
python main-master/experiments/train_seed.py -c baselines/DLinear/CA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/CA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/CA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/DLinear/CA_96_672.py -g 0