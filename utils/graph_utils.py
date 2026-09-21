import random
import pandas as pd
import numpy as np
import networkx as nx
import torch
from typing import Literal
from tqdm import tqdm

class GraphUtils:
    @staticmethod
    def remove_self_loop(graph:nx.Graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))

    @staticmethod
    def to_directed_graph(graph:nx.Graph)->nx.DiGraph:
        """
        undirected graph의 edge를 두 방향중 랜덤 한 방향만 설정
        """
        digraph=nx.DiGraph()
        digraph.add_nodes_from(graph.nodes())
        digraph.add_nodes_from(graph.nodes(data=True))
        for src,dst in graph.edges():
            if random.random()<0.5:
                digraph.add_edge(src,dst)
            else:
                digraph.add_edge(dst,src)
        return digraph

    @staticmethod
    def set_edge_timestamp(graph:nx.DiGraph):
        for edge in graph.edges():
            num_timestamps=random.randint(1,5)
            graph.edges[edge]["t"]=[
                random.randint(0,1000) # unix_timestamp (day) 의미
                for _ in range(num_timestamps)
            ]

    @staticmethod
    def convert_nx_graph_to_df(graph:nx.DiGraph)->pd.DataFrame:
        """
        Return:
            DataFrame columns:
                u   : source node id (1~N)
                i   : target node id (1~N)
                t   : timestamp, int, unix_timestamp (day)
                idx : 시간순 edge index (1~E)
        """
        edge_event_list=[]
        for src,dst,data in graph.edges(data=True):
            for t in data["t"]:
                edge_event_list.append({
                    "u":src+1,
                    "i":dst+1,
                    "t":t,
                })

        # timestamp 오름차순 정렬
        graph_df=pd.DataFrame(edge_event_list)
        graph_df=graph_df.sort_values(
            by="t",
            ascending=True
        ).reset_index(drop=True)

        # edge index: 1~E
        graph_df["idx"]=range(1,len(graph_df)+1)
        return graph_df

class GraphGenerator:
    @staticmethod
    def generate_7_type_graphs(
            n_graph:int, 
            n_node:int
        )->dict[str,list[nx.DiGraph]]:
        """
        << Generate 7-type graphs >>
        1. ladder graph
        2. 2D grid graph
        3. tree graph
        4. Erdos-Renyi graph
        5. Barabasi-Albert graph
        6. 4-community graph
        7. 4-caveman graph
        """
        ladder_graph_list=[]
        grid_graph_list=[]
        tree_graph_list=[]
        erdos_renyi_graph_list=[]
        barabasi_albert_graph_list=[]
        community_graph_list=[]
        caveman_graph_list=[]

        ### generate graph
        for _ in tqdm(range(n_graph),desc=f"Generate Graph..."):
            # 1. generate ladder graph
            if n_node%2!=0:
                raise ValueError(f"ladder graph requires an even number of nodes.")
            ladder_graph=nx.ladder_graph(n_node//2)
            GraphUtils.remove_self_loop(graph=ladder_graph)
            GraphUtils.to_directed_graph(graph=ladder_graph)
            GraphUtils.set_edge_timestamp(graph=ladder_graph)
            ladder_graph_list.append(ladder_graph)

            # 2. generate 2D grid graph
            side_length=int(np.ceil(np.sqrt(n_node)))
            grid_graph=nx.grid_2d_graph(side_length,side_length)
            grid_graph=nx.convert_node_labels_to_integers(grid_graph)
            grid_graph=grid_graph.subgraph(range(n_node)).copy()
            GraphUtils.remove_self_loop(graph=grid_graph)
            GraphUtils.to_directed_graph(graph=grid_graph)
            GraphUtils.set_edge_timestamp(graph=grid_graph)
            grid_graph_list.append(grid_graph)

            # 3. generate tree graph
            tree_graph=nx.random_labeled_tree(n_node)
            GraphUtils.remove_self_loop(graph=tree_graph)
            GraphUtils.to_directed_graph(graph=tree_graph)
            GraphUtils.set_edge_timestamp(graph=tree_graph)
            tree_graph_list.append(tree_graph)

            # 4. generate Erdos-Renyi graph
            p=min(np.log2(n_node)/n_node,0.5)
            erdos_renyi_graph=nx.erdos_renyi_graph(n_node,p)
            GraphUtils.remove_self_loop(graph=erdos_renyi_graph)
            GraphUtils.to_directed_graph(graph=erdos_renyi_graph)
            GraphUtils.set_edge_timestamp(graph=erdos_renyi_graph)
            erdos_renyi_graph_list.append(erdos_renyi_graph)

            # 5. generate Barabasi-Albert graph
            if n_node<=4:
                raise ValueError("Barabasi-Albert graph requires more than 4 number of nodes.")
            m=random.choice([4,5])
            barabasi_albert_graph=nx.barabasi_albert_graph(n_node,m)
            GraphUtils.remove_self_loop(graph=barabasi_albert_graph)
            GraphUtils.to_directed_graph(graph=barabasi_albert_graph)
            GraphUtils.set_edge_timestamp(graph=barabasi_albert_graph)
            barabasi_albert_graph_list.append(barabasi_albert_graph)

            # 6. generate 4 community graph
            if n_node<4:
                raise ValueError("4-Community graph requires at least 4 nodes.")
            community_size=n_node//4
            remaining_nodes=n_node%4
            communities=[nx.erdos_renyi_graph(community_size,0.1) for _ in range(4)]
            community_graph=nx.disjoint_union_all(communities)
            for i in range(remaining_nodes):
                community_graph.add_node(community_graph.number_of_nodes())
            nodes=list(community_graph.nodes())
            for i in range(len(nodes)):
                for j in range(i+1,len(nodes)):
                    if (i//community_size)!=(j//community_size):
                        if random.random()<0.01:
                            community_graph.add_edge(i,j)
            GraphUtils.remove_self_loop(graph=community_graph)
            GraphUtils.to_directed_graph(graph=community_graph)
            GraphUtils.set_edge_timestamp(graph=community_graph)
            community_graph_list.append(community_graph)

            # 7. generate 4-caveman graph
            if n_node<4:
                raise ValueError("4-Caveman graph requires at least 4 nodes.")
            clique_size=n_node//4
            remaining_nodes=n_node%4
            caveman_graph=nx.caveman_graph(4,clique_size)
            for i in range(remaining_nodes):
                caveman_graph.add_node(caveman_graph.number_of_nodes())
            edges_to_remove=[edge for edge in caveman_graph.edges() if random.random()<0.8]
            caveman_graph.remove_edges_from(edges_to_remove)
            num_shortcuts=int(0.025*n_node)
            for _ in range(num_shortcuts):
                u,v=random.sample(list(caveman_graph.nodes()),2)
                if not caveman_graph.has_edge(u,v):
                    caveman_graph.add_edge(u,v)
            GraphUtils.remove_self_loop(graph=caveman_graph)
            GraphUtils.to_directed_graph(graph=caveman_graph)
            GraphUtils.set_edge_timestamp(graph=caveman_graph)
            caveman_graph_list.append(caveman_graph)
        return {
            "ladder":ladder_graph_list,
            "grid":grid_graph_list,
            "tree":tree_graph_list,
            "erdos_renyi":erdos_renyi_graph_list,
            "barabasi_albert":barabasi_albert_graph_list,
            "community":community_graph_list,
            "caveman":caveman_graph_list
        }
