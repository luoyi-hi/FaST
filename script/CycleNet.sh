# CycleNet on SD dataset
python main-master/experiments/train_seed.py -c baselines/CycleNet/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/SD_96_672.py -g 0

# CycleNet on GBA dataset
python main-master/experiments/train_seed.py -c baselines/CycleNet/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/GBA_96_672.py -g 0

# CycleNet on GLA dataset
python main-master/experiments/train_seed.py -c baselines/CycleNet/GLA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/GLA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/GLA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/GLA_96_672.py -g 0

# CycleNet on CA dataset
python main-master/experiments/train_seed.py -c baselines/CycleNet/CA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/CA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/CA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/CycleNet/CA_96_672.py -g 0