cd main-master/baselines/SGP

export PYTHONPATH=$(pwd):$PYTHONPATH

# SGP on SD dataset
python experiments/run_traffic_sgps_sd_96_48.py
python experiments/run_traffic_sgps_sd_96_96.py
python experiments/run_traffic_sgps_sd_96_192.py
python experiments/run_traffic_sgps_sd_96_672.py