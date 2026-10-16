#!/bin/bash
set -e

python -m app.evaluate.evaluate_ID --dataset_name CollegeMsg

python -m app.evaluate.evaluate_ID --dataset_name bitcoin-alpha

python -m app.evaluate.evaluate_ID --dataset_name bitcoin-otc