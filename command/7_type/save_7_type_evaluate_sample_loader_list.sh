#!/bin/bash
set -e

python -m app.7_type.save_7_type_evaluate_sample_loader_list --n_node 100 --batch_size 32 --purpose val
python -m app.7_type.save_7_type_evaluate_sample_loader_list --n_node 100 --batch_size 32 --purpose test
python -m app.7_type.save_7_type_evaluate_sample_loader_list --n_node 500 --batch_size 200 --purpose test
python -m app.7_type.save_7_type_evaluate_sample_loader_list --n_node 1000 --batch_size 200 --purpose test