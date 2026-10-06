#!/bin/bash
set -e

python -m app.save_TR_result --dataset_name CollegeMsg --purpose train --batch_size 200
python -m app.save_TR_result --dataset_name CollegeMsg --purpose val --batch_size 200
python -m app.save_TR_result --dataset_name CollegeMsg --purpose test --batch_size 200