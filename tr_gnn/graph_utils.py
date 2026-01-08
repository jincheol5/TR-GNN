import random
import numpy as np
import networkx as nx
import torch
from typing_extensions import Literal
from tqdm import tqdm

class GraphGenerator:
    @staticmethod
    def set_edge_time_attr(graph:nx.DiGraph,num_times:int):
        for src,tar in graph.edges():
            if src==tar:
                graph[src][tar]['t']=[1.1]
            else:
                timestamp_num=np.random.randint(1,num_times+1)
                timestamps=np.random.uniform(0.2,1.0,size=timestamp_num)
                timestamps=np.sort(timestamps).tolist() 
                graph[src][tar]['t']=timestamps

    @staticmethod
    def remove_self_loop(graph:nx.Graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))

    @staticmethod
    def remove_random_edges(graph:nx.DiGraph,ratio:float=0.3):
        edges=[(u,v) for u,v in graph.edges() if u!=v]
        num_edges=len(edges)
        num_remove=int(num_edges*ratio)
        remove_edges=random.sample(edges,num_remove)
        graph.remove_edges_from(remove_edges)
    
    @staticmethod
    def generate_7_type_graph(graph_type:Literal['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman'],num_nodes:Literal[20,50,100,500,1000],num_times:int=5):
        """
        <<Generate 7-type graphs>>
        1. ladder graph
        2. 2D grid graph
        3. tree graph
        4. Erdos-Renyi graph
        5. Barabasi-Albert graph
        6. 4-community graph
        7. 4-caveman graph
        """
        match graph_type:
            case 'ladder':
                if num_nodes%2!=0:
                    raise ValueError("ladder graph requires an even number of nodes.")
                graph=nx.ladder_graph(num_nodes//2)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'grid':
                side_length=int(np.ceil(np.sqrt(num_nodes)))
                graph=nx.grid_2d_graph(side_length,side_length)
                graph=nx.convert_node_labels_to_integers(graph)
                graph=graph.subgraph(range(num_nodes)).copy()
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'tree':
                graph=nx.random_tree(num_nodes)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'erdos_renyi':
                p=min(np.log2(num_nodes)/num_nodes,0.5)
                graph=nx.erdos_renyi_graph(num_nodes,p)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                if num_nodes>=500:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                else:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.5)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'barabasi_albert':
                if num_nodes<=4:
                    raise ValueError("barabasi_albert graph requires more than 4 number of nodes.")
                m=random.choice([4,5])
                graph=nx.barabasi_albert_graph(num_nodes,m)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'community':
                if num_nodes<4:
                    raise ValueError("4-Community graph requires at least 4 nodes.")
                community_size=num_nodes//4
                remaining_nodes=num_nodes%4
                communities=[nx.erdos_renyi_graph(community_size,0.1) for _ in range(4)]
                graph=nx.disjoint_union_all(communities)
                for i in range(remaining_nodes):
                    graph.add_node(graph.number_of_nodes())
                nodes=list(graph.nodes())
                for i in range(len(nodes)):
                    for j in range(i+1,len(nodes)):
                        if (i//community_size)!=(j//community_size):
                            if random.random()<0.01:
                                graph.add_edge(i,j)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                if num_nodes>=500:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'caveman':
                if num_nodes<4:
                    raise ValueError("4-Caveman graph requires at least 4 nodes.")
                clique_size=num_nodes//4
                remaining_nodes=num_nodes%4
                graph=nx.caveman_graph(4,clique_size)
                for i in range(remaining_nodes):
                    graph.add_node(graph.number_of_nodes())
                edges_to_remove=[edge for edge in graph.edges() if random.random()<0.8]
                graph.remove_edges_from(edges_to_remove)
                num_shortcuts=int(0.025*num_nodes)
                for _ in range(num_shortcuts):
                    u,v=random.sample(list(graph.nodes()),2)
                    if not graph.has_edge(u,v):
                        graph.add_edge(u,v)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                if num_nodes>=500:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
        return graph

    @staticmethod
    def generate_7_type_graph_list(num_graphs:int,graph_type:Literal['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman'],num_nodes:Literal[20,50,100,500,1000],num_times:int=5):
        graph_list=[]
        for _ in tqdm(range(num_graphs),desc=f"generate graph..."):
            graph_list.append(GraphGenerator.generate_7_type_graph(graph_type=graph_type,num_nodes=num_nodes,num_times=num_times))
        return graph_list

class GraphUtils:
    @staticmethod
    def get_eventstream(graph:nx.DiGraph):
        """
        Input:
            networkx DiGraph
        Output:
            sorted tuple list: (src,tar,ts)
        """
        eventstream=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['t']
            for timestamp in time_list:
                eventstream.append((int(u),int(v),float(timestamp)))
        eventstream=sorted(eventstream,key=lambda x:x[2])
        return eventstream
    
    @staticmethod
    def compute_datastream_from_eventstream(eventstream:list,num_nodes:int):
        """
        Input:
            event_stream: sorted tuple list (src,tar,ts)
            num_nodes: number of nodes
        Output:
            datastream: dict 
                mem_t: [E,N,1], delta_t for memory update -> |current_time-previous_activated_time (with any nodes)| of each node
                emb_t: [E,N,1], delta_t for embedding -> |current_time-previous_interaction_time (with target node)| of target node
                src: [E,1], source id
                tar: [E,1], target id
                n_mask: [E,N], neighbor mask of target node
        """

        # initial setup for delta_t in memory update, embedding
        mem_time_table=torch.zeros((num_nodes,),dtype=torch.float) # last activated time of node
        emb_time_table=torch.zeros((num_nodes,num_nodes),dtype=torch.float) # interaction time from src to tar
        
        # initial setup for n_mask
        num_edge_events=len(eventstream)
        neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool) # [E,N], 각 edge_event에 대한 tar의 neighbor mask
        neighbor_history=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)] # List of [N,]

        src_list=[]
        tar_list=[]
        mem_t_list=[]
        emb_t_list=[]
        n_mask_list=[]
        for idx,edge_event in enumerate(eventstream):
            src,tar,ts=edge_event
            src_list.append(src)
            tar_list.append(tar)

            emb_time_table[tar,src]=ts
            interacted_t=emb_time_table[tar].unsqueeze(-1) # [N,1]
            emb_t_list.append(torch.abs(interacted_t-ts))

            activated_t=mem_time_table.unsqueeze(-1) # [N,1]
            mem_t_list.append(torch.abs(activated_t-ts))
            mem_time_table[src]=ts
            mem_time_table[tar]=ts

            neighbor_history[tar][src]=True
            neighbor_mask[idx]=neighbor_history[tar] # 참조가 아닌 복사(tensor index 대입)
            n_mask_list.append(neighbor_mask[idx])

        # convert to datastream
        datastream={}
        datastream['src']=torch.tensor(src_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        datastream['tar']=torch.tensor(tar_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        datastream['mem_t']=torch.stack(mem_t_list,dim=0) # [E,N,1]
        datastream['emb_t']=torch.stack(emb_t_list,dim=0) # [E,N,1]
        datastream['n_mask']=torch.stack(n_mask_list,dim=0) # [E,N]
        return datastream

    @staticmethod
    def compute_TR_step(num_nodes:int,source_id:int,edge_event:tuple=None,init:bool=False,gamma:torch.Tensor=None):
        """
        Input:
            source_id
            edge_event
            init
            gamma
        Output:
            gamma
        """
        if init:
            gamma=torch.zeros((num_nodes,2),dtype=torch.float) # TR,visited_time
            gamma[:,0]=0.0 # TR
            gamma[:,1]=1.1 # visited time

            gamma[source_id,0]=1.0
            gamma[source_id,1]=0.0
        else:
            src,tar,ts=edge_event
            if gamma[src,0].item()==1.0 and gamma[tar,0].item()==0.0 and gamma[src,1].item()<ts:
                gamma[tar,0]=1.0
                gamma[tar,1]=ts
        return gamma

    @staticmethod
    def compute_TR_trajectory_from_eventstream(eventstream:list,num_nodes:int,source_id:int):
        """
        Input:
            event_stream: sorted tuple list (src,tar,ts)
            num_nodes: number of nodes
            source_id: source node id
        Output:
            TR_trajectory
        """
        TR_list=[]
        gamma=GraphUtils.compute_TR_step(num_nodes=num_nodes,source_id=source_id,init=True)
        for edge_event in eventstream:
            gamma=GraphUtils.compute_TR_step(
                num_nodes=num_nodes,
                source_id=source_id,
                edge_event=edge_event,
                gamma=gamma
            )
            TR_list.append(gamma[:,:1])
        TR_trajectory=torch.stack(TR_list,dim=0) # [E,N,1]
        return TR_trajectory

    @staticmethod
    def convert_graph_to_dataset(graph:nx.DiGraph,src_list:list):
        """
        Input:
            graph: networkx DiGraph
            src_list: source node list
        Output:
            dataset: dict
                src_list
                datastream
                traj_list
        """
        num_nodes=graph.number_of_nodes()
        eventstream=GraphUtils.get_eventstream(graph=graph)
        datastream=GraphUtils.compute_datastream_from_eventstream(eventstream=eventstream,num_nodes=num_nodes) # dict
        
        traj_list=[]
        for source_id in tqdm(src_list,desc=f"Converting graph to dataset..."):
            traj=GraphUtils.compute_TR_trajectory_from_eventstream(eventstream=eventstream,num_nodes=num_nodes,source_id=source_id) # [E,N,1]
            traj_list.append(traj)
        
        return {
            'src_list':src_list,
            'datastream':datastream,
            'traj_list':traj_list # list of [E,N,1]
        }

class GraphAnalysis:
    def check_elements(graph:nx.DiGraph):
        num_nodes=graph.number_of_nodes()
        num_static_edges=graph.number_of_edges()
        num_edge_events=0
        for src,tar in graph.edges():
            count=0
            for time in graph[src][tar]['t']:
                if time!=0.0:
                    count+=1
            num_edge_events+=count
        return num_nodes,num_static_edges,num_edge_events
    
    @staticmethod
    def check_tR_ratio(r:torch.Tensor): 
        r_flat=r.view(-1).to(torch.float32) # [N,1] -> [N,] 
        return r_flat.mean().item()