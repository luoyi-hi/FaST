# NHiTS on SD dataset
python main-master/experiments/train_seed.py -c baselines/NHiTS/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/SD_96_672.py -g 0

# NHiTS on GBA dataset
python main-master/experiments/train_seed.py -c baselines/NHiTS/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/GBA_96_672.py -g 0

# NHiTS on GLA dataset
python main-master/experiments/train_seed.py -c baselines/NHiTS/GLA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/GLA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/GLA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/GLA_96_672.py -g 0

# NHiTS on CA dataset
python main-master/experiments/train_seed.py -c baselines/NHiTS/CA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/CA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/CA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/NHiTS/CA_96_672.py -g 0