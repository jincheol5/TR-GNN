#!/bin/bash
set -e

python -m app.SNAP.save_TR_result --dataset_name CollegeMsg --batch_size 200
