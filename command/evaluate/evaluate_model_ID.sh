#!/bin/bash
set -e

python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type base

python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base

python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type hop_range

python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range



python -m app.evaluate.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range