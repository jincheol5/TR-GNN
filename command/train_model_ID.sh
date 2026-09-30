#!/bin/bash
set -e

python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg --save_model 1
python -m app.train_model_ID --model_name TGN --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg --save_model 1

python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0005 --sampling dependent --dataset_name CollegeMsg --source 418 --save_model 1
python -m app.train_model_ID --model_name TGN --seed 1 --lr 0.0005 --sampling dependent --dataset_name CollegeMsg --source 418 --save_model 1

