#!/bin/bash
set -e

python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg 
python -m app.train_model_ID --model_name TGN --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg 
python -m app.train_model_ID --model_name DyGFormer --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg 
python -m app.train_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg 

python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0005 --sampling dependent --dataset_name CollegeMsg --source 418
python -m app.train_model_ID --model_name TGN --seed 1 --lr 0.0005 --sampling dependent --dataset_name CollegeMsg --source 418
python -m app.train_model_ID --model_name DyGFormer --seed 1 --lr 0.0005 --sampling dependent --dataset_name CollegeMsg --source 418
python -m app.train_model_ID --model_name ReaCH-TGN --seed 1 --lr 0.0005 --sampling dependent --dataset_name CollegeMsg --source 418
