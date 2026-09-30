#!/bin/bash
set -e

python -m app.find_source_candidates --dataset_name CollegeMsg
python -m app.find_source_candidates --dataset_name bitcoin-alpha 
python -m app.find_source_candidates --dataset_name bitcoin-otc 