#!/bin/bash
set -e

python -m app.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --source 259
python -m app.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --source 259

python -m app.evaluate_model_ID --model_name TGAT --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --source 259
python -m app.evaluate_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --source 259