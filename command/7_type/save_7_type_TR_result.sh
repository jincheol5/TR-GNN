#!/bin/bash
set -e

# for graph_type in ladder grid tree erdos_renyi barabasi_albert community caveman; do
#     for purpose in train val test; do
#         if [ "$purpose" = "test" ]; then
#             n_nodes="100 500 1000"
#         else
#             n_nodes="100"
#         fi

#         for n_node in $n_nodes; do
#             if [ "$n_node" -eq 100 ]; then
#                 batch_size=32
#             else
#                 batch_size=200
#             fi

#             python -m app.7_type.save_7_type_TR_result --graph_type "$graph_type" --purpose "$purpose" --n_node "$n_node" --batch_size "$batch_size"
#         done
#     done
# done

for graph_type in ladder grid tree erdos_renyi barabasi_albert community caveman; do
    for purpose in train; do
        if [ "$purpose" = "test" ]; then
            n_nodes="100 500 1000"
        else
            n_nodes="100"
        fi

        for n_node in $n_nodes; do
            if [ "$n_node" -eq 100 ]; then
                batch_size=32
            else
                batch_size=200
            fi

            python -m app.7_type.save_7_type_TR_result --graph_type "$graph_type" --purpose "$purpose" --n_node "$n_node" --batch_size "$batch_size"
        done
    done
done