#!/bin/bash
set -e

python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1
python -m app.train_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1
python -m app.train_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0001 --sampling random --dataset_name CollegeMsg --save_model 1


python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --source 259 --save_model 1
python -m app.train_model_ID --model_name TGN --seed 1 --lr 0.0001 --sampling focused --dataset_name CollegeMsg --source 259 --save_model 1

