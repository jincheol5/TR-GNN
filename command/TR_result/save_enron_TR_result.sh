#!/bin/bash
set -e

python -m app.save_TR_result --dataset_name enron --purpose train --batch_size 200
python -m app.save_TR_result --dataset_name enron --purpose val --batch_size 200
python -m app.save_TR_result --dataset_name enron --purpose test --batch_size 200