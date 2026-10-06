#!/bin/bash
set -e

python -m app.save_TR_result --dataset_name bitcoin-alpha --purpose train --batch_size 200
python -m app.save_TR_result --dataset_name bitcoin-alpha --purpose val --batch_size 200
python -m app.save_TR_result --dataset_name bitcoin-alpha --purpose test --batch_size 200