#!/bin/bash
set -e

python -m app.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --evaluate_type base --source 259
python -m app.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --evaluate_type base --source 259

python -m app.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --evaluate_type base --source 259
python -m app.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --evaluate_type base --source 259
python -m app.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --evaluate_type base --source 259


python -m app.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --evaluate_type hop_range --source 259

python -m app.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --evaluate_type hop_range --source 259
python -m app.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --evaluate_type hop_range --source 259