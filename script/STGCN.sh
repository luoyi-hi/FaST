# STGCN on SD dataset
python main-master/experiments/train_seed.py -c baselines/STGCN/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STGCN/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STGCN/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STGCN/SD_96_672.py -g 0

# STGCN on GBA dataset
python main-master/experiments/train_seed.py -c baselines/STGCN/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STGCN/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STGCN/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STGCN/GBA_96_672.py -g 0
