# STID on SD dataset
python main-master/experiments/train_seed.py -c baselines/STID/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/SD_96_672.py -g 0

# STID on GBA dataset
python main-master/experiments/train_seed.py -c baselines/STID/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/GBA_96_672.py -g 0

# STID on GLA dataset
python main-master/experiments/train_seed.py -c baselines/STID/GLA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/GLA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/GLA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/GLA_96_672.py -g 0

# STID on CA dataset
python main-master/experiments/train_seed.py -c baselines/STID/CA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/CA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/CA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STID/CA_96_672.py -g 0