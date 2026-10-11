#!/bin/bash
set -e

python -m app.generate_incomplete_noisy --dataset_name CollegeMsg --batch_size 200
python -m app.generate_incomplete_noisy --dataset_name bitcoin-alpha --batch_size 200
python -m app.generate_incomplete_noisy --dataset_name bitcoin-otc --batch_size 200