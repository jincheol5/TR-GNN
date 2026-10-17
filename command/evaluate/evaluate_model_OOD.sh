#!/bin/bash
set -e

# TGAT
python -m app.evaluate.evaluate_model_OOD --model_name TGAT --n_node 100 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name TGAT --n_node 500 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name TGAT --n_node 1000 --seed 1 --lr 0.0001 

# TGN
python -m app.evaluate.evaluate_model_OOD --model_name TGN --n_node 100 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name TGN --n_node 500 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name TGN --n_node 1000 --seed 1 --lr 0.0001 

# DyGFormer
python -m app.evaluate.evaluate_model_OOD --model_name DyGFormer --n_node 100 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name DyGFormer --n_node 500 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name DyGFormer --n_node 1000 --seed 1 --lr 0.0001 

# ReaCH-TGN
python -m app.evaluate.evaluate_model_OOD --model_name ReaCH-TGN --n_node 100 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name ReaCH-TGN --n_node 500 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name ReaCH-TGN --n_node 1000 --seed 1 --lr 0.0001 

# TR-GNN
python -m app.evaluate.evaluate_model_OOD --model_name TR-GNN --n_node 100 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name TR-GNN --n_node 500 --seed 1 --lr 0.0001 
python -m app.evaluate.evaluate_model_OOD --model_name TR-GNN --n_node 1000 --seed 1 --lr 0.0001 