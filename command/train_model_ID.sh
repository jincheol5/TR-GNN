#!/bin/bash
set -e

python -m app.train_model_ID --model_name TGAT --seed 1 --lr 0.0005 --sampling independent --dataset_name CollegeMsg --source 