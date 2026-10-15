#!/bin/bash
set -e

# TGAT
python -m app.evaluate.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type hop_range
python -m app.evaluate.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range

# TGN
python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type hop_range
python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range

# DyGFormer
python -m app.evaluate.evaluate_model_ID --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type hop_range
python -m app.evaluate.evaluate_model_ID --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range

# ReaCH-TGN
python -m app.evaluate.evaluate_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random --evaluate_type hop_range
python -m app.evaluate.evaluate_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range

# TR-GNN
python -m app.evaluate.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type base
python -m app.evaluate.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused --evaluate_type hop_range