python DataPipeline/generate_data.py

python DataPipeline/generate_data_for_training.py --dataset sd --years 2019
python DataPipeline/generate_data_for_training.py --dataset gba --years 2019
python DataPipeline/generate_data_for_training.py --dataset gla --years 2019
python DataPipeline/generate_data_for_training.py --dataset ca --years 2019

python DataPipeline/process_adj.py

python DataPipeline/generate_idx.py