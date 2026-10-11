#!/bin/bash
set -e

python -m app.save_TR_result --dataset_name CollegeMsg --batch_size 200

python -m app.save_TR_result --dataset_name bitcoin-alpha --batch_size 200

python -m app.save_TR_result --dataset_name bitcoin-otc --batch_size 200
