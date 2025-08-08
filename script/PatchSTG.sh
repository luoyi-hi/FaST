# PatchSTG on SD dataset
python main-master/experiments/train_seed.py -c baselines/PatchSTG/SD_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/SD_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/SD_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/SD_96_672.py -g 0

# PatchSTG on GBA dataset
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GBA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GBA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GBA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GBA_96_672.py -g 0

# PatchSTG on GLA dataset
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GLA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GLA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GLA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/GLA_96_672.py -g 0

# PatchSTG on CA dataset
python main-master/experiments/train_seed.py -c baselines/PatchSTG/CA_96_48.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/CA_96_96.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/CA_96_192.py -g 0
python main-master/experiments/train_seed.py -c baselines/PatchSTG/CA_96_672.py -g 0