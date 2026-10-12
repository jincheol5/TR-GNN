#!/bin/bash
set -e

for graph_type in ladder grid tree erdos_renyi barabasi_albert community caveman; do
    for purpose in train val test; do
        for n_node in 100 500 1000; do
            if [ "$n_node" -eq 100 ]; then
                batch_size=32
            else
                batch_size=200
            fi

            python -m app.main --graph_type "$graph_type" --purpose "$purpose" --n_node "$n_node" --batch_size "$batch_size"
        done
    done
done