#!/bin/bash
set -e

# TGAT
python -m app.evaluate.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random 
python -m app.evaluate.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused 

# TGN
python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random 
python -m app.evaluate.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused 

# DyGFormer
python -m app.evaluate.evaluate_model_ID --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random 
python -m app.evaluate.evaluate_model_ID --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused 

# ReaCH-TGN
python -m app.evaluate.evaluate_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling random 
python -m app.evaluate.evaluate_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused 

# TR-GNN
python -m app.evaluate.evaluate_model_ID --model_name TR-GNN --seed 1 --lr 0.0001 --dataset_name CollegeMsg --source 259 --sampling focused 
