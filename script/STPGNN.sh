# STPGNN on SD dataset
python main-master/experiments/train_seed.py -c baselines/STPGNN/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STPGNN/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STPGNN/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STPGNN/SD_96_672.py -g 0

