#!/bin/bash

# for model_name in TGAT TGN DyGFormer ReaCH-TGN TR-GNN; do
#     for seed in 1 2 3; do
#         for lr in 0.0005 0.0001; do
#             python -m app.train.train_model_OOD --model_name "$model_name" --seed "$seed" --lr "$lr" --save_model 1
#         done
#     done
# done


python -m app.train.train_model_OOD --model_name TGN --seed 3 --lr 0.0001 --save_model 1
python -m app.train.train_model_OOD --model_name TGN --seed 3 --lr 0.0005 --save_model 1

for model_name in DyGFormer ReaCH-TGN TR-GNN; do
    for seed in 1 2 3; do
        for lr in 0.0005 0.0001; do
            python -m app.train.train_model_OOD --model_name "$model_name" --seed "$seed" --lr "$lr" --save_model 1
        done
    done
done