#!/bin/bash

# CollegeMsg
for source in 259 319 673 86 274 711 272 554 617 439; do
    for model_name in TGAT TGN DyGFormer ReaCH-TGN; do
        for seed in 1 2 3; do
            for lr in 0.0005 0.0001; do
                python -m app.train.train_model_ID --model_name "$model_name" --seed "$seed" --lr "$lr" --dataset_name CollegeMsg --source "$source" --sampling random --save_model 1
            done
        done
    done
done

for source in 259 319 673 86 274 711 272 554 617 439; do
    for model_name in TGAT TGN DyGFormer ReaCH-TGN TR-GNN; do
        for seed in 1 2 3; do
            for lr in 0.0005 0.0001; do
                python -m app.train.train_model_ID --model_name "$model_name" --seed "$seed" --lr "$lr" --dataset_name CollegeMsg --source "$source" --sampling focused --save_model 1
            done
        done
    done
done

# bitcoin-alpha
for source in 323 566 402 717 1402 1216 1212 440 1928 299; do
    for model_name in TGAT TGN DyGFormer ReaCH-TGN; do
        for seed in 1 2 3; do
            for lr in 0.0005 0.0001; do
                python -m app.train.train_model_ID --model_name "$model_name" --seed "$seed" --lr "$lr" --dataset_name bitcoin-alpha --source "$source" --sampling random --save_model 1
            done
        done
    done
done

for source in 323 566 402 717 1402 1216 1212 440 1928 299; do
    for model_name in TGAT TGN DyGFormer ReaCH-TGN TR-GNN; do
        for seed in 1 2 3; do
            for lr in 0.0005 0.0001; do
                python -m app.train.train_model_ID --model_name "$model_name" --seed "$seed" --lr "$lr" --dataset_name bitcoin-alpha --source "$source" --sampling focused --save_model 1
            done
        done
    done
done

# bitcoin-otc
for source in 89 273 212 198 42 57 107 147 344 45; do
    for model_name in TGAT TGN DyGFormer ReaCH-TGN; do
        for seed in 1 2 3; do
            for lr in 0.0005 0.0001; do
                python -m app.train.train_model_ID --model_name "$model_name" --seed "$seed" --lr "$lr" --dataset_name bitcoin-otc --source "$source" --sampling random --save_model 1
            done
        done
    done
done

for source in 89 273 212 198 42 57 107 147 344 45; do
    for model_name in TGAT TGN DyGFormer ReaCH-TGN TR-GNN; do
        for seed in 1 2 3; do
            for lr in 0.0005 0.0001; do
                python -m app.train.train_model_ID --model_name "$model_name" --seed "$seed" --lr "$lr" --dataset_name bitcoin-otc --source "$source" --sampling focused --save_model 1
            done
        done
    done
done
