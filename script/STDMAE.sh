# STDMAE on SD dataset
python main-master/experiments/train_seed.py -c baselines/STDMAE/S-SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/S-SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/S-SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/S-SD_96_672.py -g 0

python main-master/experiments/train_seed.py -c baselines/STDMAE/T-SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/T-SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/T-SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/T-SD_96_672.py -g 0

python main-master/experiments/train_seed.py -c baselines/STDMAE/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/STDMAE/SD_96_672.py -g 0