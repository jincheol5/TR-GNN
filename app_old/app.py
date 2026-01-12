import networkx as nx
from trgnn import DataUtils 

graph_list_dict=DataUtils.load_from_pickle(file_name=f"test_1000",dir_type="graph",num_nodes=1000)

total_sum=0
for graph_type,graph_list in graph_list_dict.items():
    for graph in graph_list:
        has_self_loop=any(nx.selfloop_edges(graph))
        if has_self_loop:
            total_sum+=1
print(f"total_sum: {total_sum}")