import random
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

class GraphGenerator:
    @staticmethod
    def remove_self_loop(graph:nx.Graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))

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
    def remove_random_edges(graph:nx.DiGraph,ratio:float=0.3):
        edges=[(u,v) for u,v in graph.edges() if u!=v]
        num_edges=len(edges)
        num_remove=int(num_edges*ratio)
        remove_edges=random.sample(edges,num_remove)
        graph.remove_edges_from(remove_edges)
    
    @staticmethod
    def generate_7_type_graphs(num_graphs:int,num_nodes:int,num_times:int):
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

        ladder_graph_list=[]
        grid_graph_list=[]
        tree_graph_list=[]
        Erdos_Renyi_graph_list=[]
        Barabasi_Albert_graph_list=[]
        community_graph_list=[]
        caveman_graph_list=[]

        # generate
        for _ in tqdm(range(num_graphs),desc=f"generate graph..."):
            """
            1. generate ladder graph
            """
            if num_nodes%2!=0:
                raise ValueError("ladder graph requires an even number of nodes.")
            ladder_graph=nx.ladder_graph(num_nodes//2)
            GraphGenerator.remove_self_loop(graph=ladder_graph)
            ladder_graph=ladder_graph.to_directed()
            GraphGenerator.set_edge_time_attr(graph=ladder_graph,num_times=num_times)
            ladder_graph_list.append(ladder_graph)

            """
            2. generate 2D grid graph
            """
            side_length=int(np.ceil(np.sqrt(num_nodes)))
            grid_graph=nx.grid_2d_graph(side_length,side_length)
            grid_graph=nx.convert_node_labels_to_integers(grid_graph)
            grid_graph=grid_graph.subgraph(range(num_nodes)).copy()
            GraphGenerator.remove_self_loop(graph=grid_graph)
            grid_graph=grid_graph.to_directed()
            GraphGenerator.set_edge_time_attr(graph=grid_graph,num_times=num_times)
            grid_graph_list.append(grid_graph)

            
            """
            3. generate tree graph
            """
            tree_graph=nx.random_tree(num_nodes)
            GraphGenerator.remove_self_loop(graph=tree_graph)
            tree_graph=tree_graph.to_directed()
            GraphGenerator.set_edge_time_attr(graph=tree_graph,num_times=num_times)
            tree_graph_list.append(tree_graph)

            """
            4. generate Erdos-Renyi graph
            """
            p=min(np.log2(num_nodes)/num_nodes,0.5)
            erdos_renyi_graph=nx.erdos_renyi_graph(num_nodes,p)
            GraphGenerator.remove_self_loop(graph=erdos_renyi_graph)
            erdos_renyi_graph=erdos_renyi_graph.to_directed()
            if num_nodes>=500:
                GraphGenerator.remove_random_edges(graph=erdos_renyi_graph,ratio=0.7)
            else:
                GraphGenerator.remove_random_edges(graph=erdos_renyi_graph,ratio=0.5)
            GraphGenerator.set_edge_time_attr(graph=erdos_renyi_graph,num_times=num_times)
            Erdos_Renyi_graph_list.append(erdos_renyi_graph)

            """
            5. generate Barabasi-Albert graph
            """
            if num_nodes<=4:
                raise ValueError("barabasi_albert graph requires more than 4 number of nodes.")
            m=random.choice([4,5])
            barabasi_albert_graph=nx.barabasi_albert_graph(num_nodes,m)
            GraphGenerator.remove_self_loop(graph=barabasi_albert_graph)
            barabasi_albert_graph=barabasi_albert_graph.to_directed()
            GraphGenerator.remove_random_edges(graph=barabasi_albert_graph,ratio=0.7)
            GraphGenerator.set_edge_time_attr(graph=barabasi_albert_graph,num_times=num_times)
            Barabasi_Albert_graph_list.append(barabasi_albert_graph)
            
            """
            6. generate 4 community graph
            """
            if num_nodes<4:
                raise ValueError("4-Community graph requires at least 4 nodes.")
            community_size=num_nodes//4
            remaining_nodes=num_nodes%4
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
            GraphGenerator.remove_self_loop(graph=community_graph)
            community_graph=community_graph.to_directed()
            if num_nodes>=500:
                GraphGenerator.remove_random_edges(graph=community_graph,ratio=0.7)
            GraphGenerator.set_edge_time_attr(graph=community_graph,num_times=num_times)
            community_graph_list.append(community_graph)
        
            """
            7. generate 4-caveman graph
            """
            if num_nodes<4:
                raise ValueError("4-Caveman graph requires at least 4 nodes.")
            clique_size=num_nodes//4
            remaining_nodes=num_nodes%4
            caveman_graph=nx.caveman_graph(4,clique_size)
            for i in range(remaining_nodes):
                caveman_graph.add_node(caveman_graph.number_of_nodes())
            edges_to_remove=[edge for edge in caveman_graph.edges() if random.random()<0.8]
            caveman_graph.remove_edges_from(edges_to_remove)
            num_shortcuts=int(0.025*num_nodes)
            for _ in range(num_shortcuts):
                u,v=random.sample(list(caveman_graph.nodes()),2)
                if not caveman_graph.has_edge(u,v):
                    caveman_graph.add_edge(u,v)
            GraphGenerator.remove_self_loop(graph=caveman_graph)
            caveman_graph=caveman_graph.to_directed()
            if num_nodes>=500:
                GraphGenerator.remove_random_edges(graph=caveman_graph,ratio=0.7)
            GraphGenerator.set_edge_time_attr(graph=caveman_graph,num_times=num_times)
            caveman_graph_list.append(caveman_graph)

        graph_list_dict={}
        graph_list_dict['ladder']=ladder_graph_list
        graph_list_dict['grid']=grid_graph_list
        graph_list_dict['tree']=tree_graph_list
        graph_list_dict['erdos_renyi']=Erdos_Renyi_graph_list
        graph_list_dict['barabasi_albert']=Barabasi_Albert_graph_list
        graph_list_dict['community']=community_graph_list
        graph_list_dict['caveman']=caveman_graph_list
        return graph_list_dict

class GraphUtils:
    @staticmethod
    def get_event_stream(graph:nx.DiGraph):
        """
        Output:
            sorted tuple list: (src,tar,ts)
        """
        event_stream=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['t']
            for timestamp in time_list:
                event_stream.append((int(u),int(v),float(timestamp)))
        event_stream=sorted(event_stream,key=lambda x:x[2])
        return event_stream

    @staticmethod
    def compute_tR_step(num_nodes:int,source_id:int,edge_event:tuple=None,init:bool=False,gamma:torch.Tensor=None,emb_time_table:torch.Tensor=None,mem_time_table:torch.Tensor=None):
        """
        edge_event: tuple, (src,tar,ts)
        gamma: [N,2] tensor, (tR,visited time)
        emb_time_table: [N,N,] tensor, (tar_id,src_id,), value=interaction time from src to tar
        mem_time_table: [N,] tensor, (node id,). value=last activated time of node
        """
        if init:
            gamma=torch.zeros((num_nodes,3),dtype=torch.float) # tR,time,ts
            gamma[:,0]=0.0 # tR
            gamma[:,1]=0.0 # visited time

            gamma[source_id,0]=1.0
            gamma[source_id,1]=0.0

            emb_time_table=torch.zeros((num_nodes,num_nodes),dtype=torch.float)
            mem_time_table=torch.zeros((num_nodes,),dtype=torch.float)
        else:
            src,tar,ts=edge_event
            if gamma[src,0].item()==1.0 and gamma[tar,0].item()==0.0 and gamma[src,1].item()<ts:
                gamma[tar,0]=1.0
                gamma[tar,1]=ts
            emb_time_table[tar,src]=ts 
            mem_time_table[src]=ts
            mem_time_table[tar]=ts
        return gamma,emb_time_table,mem_time_table


    @staticmethod
    def convert_event_stream_to_dataset(event_stream:list,num_nodes:int,source_id:int):
        """
        Input:
            event_stream: List of edge_event tuple
            num_nodes: number of nodes
            source_id: source node id
        Output:
            dataset: sequence of data
                data:
                    raw: [N,1]
                    r: [N,1]
                    t: [N,1]
                    src: src id
                    tar: tar id
                    label: tar label, 1.0 or 0.0
                    n_mask: [N,]
        """
        raw=torch.zeros(num_nodes,1,dtype=torch.float32) # [N,1]
        raw[source_id]=1.0

        r_list=[]
        emb_t_list=[]
        mem_t_list=[]
        src_list=[]
        tar_list=[]
        label_list=[]
        n_mask_list=[]

        num_edge_events=len(event_stream)
        neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool) # [E,N], 각 edge_event에 대한 tar의 neighbor mask
        neighbor_history=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)] # List of [N,]
        gamma,emb_time_table,mem_time_table=GraphUtils.compute_tR_step(num_nodes=num_nodes,source_id=source_id,init=True) # [N,2]
        for i,edge_event in enumerate(event_stream):
            pre_mem_time_table=emb_time_table.clone()

            # compute tR step
            gamma,emb_time_table,mem_time_table=GraphUtils.compute_tR_step(num_nodes=num_nodes,source_id=source_id,edge_event=edge_event,gamma=gamma,time_table=time_table)
            
            src,tar,ts=edge_event
            r_list.append(gamma[:,:1])

            interacted_t=emb_time_table[tar].unsqueeze(-1) # [N,1]
            emb_t_list.append(torch.abs(interacted_t-ts))

            activated_t=pre_mem_time_table.unsqueeze(-1) # [N,1]
            mem_t_list.append(torch.abs(activated_t-ts))

            src_list.append(src)
            tar_list.append(tar)
            label_list.append(gamma[tar,0].item())

            neighbor_history[tar][src]=True
            neighbor_mask[i]=neighbor_history[tar] # 참조가 아닌 복사(tensor index 대입)
            n_mask_list.append(neighbor_mask[i])
        
        # convert to dataset, E = number of edge_events = seq_len
        dataset={}
        dataset['raw']=raw.unsqueeze(0).expand(num_edge_events,num_nodes,1) # [E,N,1]
        dataset['r']=torch.stack(r_list,dim=0) # [E,N,1]
        dataset['emb_t']=torch.stack(emb_t_list,dim=0) # [E,N,1]
        dataset['mem_t']=torch.stack(mem_t_list,dim=0) # [E,N,1]
        dataset['src']=torch.tensor(src_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        dataset['tar']=torch.tensor(tar_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        dataset['label']=torch.tensor(label_list,dtype=torch.float32).unsqueeze(-1) # [E,1]
        dataset['n_mask']=torch.stack(n_mask_list,dim=0) # [E,N]
        return dataset

class GraphAnalysis:
    @staticmethod
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