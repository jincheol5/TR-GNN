#!/bin/bash
set -e

for source in 259 319 673 86 274 711 272 554 617 439; do
    for evaluate_type in base hop_range; do
        python -m app.SNAP.save_test_sample_loader --dataset_name CollegeMsg --source "$source" --batch_size 200 --evaluate_type "$evaluate_type"
    done
done

for source in 323 566 402 717 1402 1216 1212 440 1928 299; do
    for evaluate_type in base hop_range; do
        python -m app.SNAP.save_test_sample_loader --dataset_name bitcoin-alpha --source "$source" --batch_size 200 --evaluate_type "$evaluate_type"
    done
done

for source in 89 273 212 198 42 57 107 147 344 45; do
    for evaluate_type in base hop_range; do
        python -m app.SNAP.save_test_sample_loader --dataset_name bitcoin-otc --source "$source" --batch_size 200 --evaluate_type "$evaluate_type"
    done
done