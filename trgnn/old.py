# @staticmethod
# def save_graph_list_to_dataset_list(graph_list:list,graph_type:str,num_nodes:int,dir_type:Literal['train','val','test']):
#     dataset_list=[]
#     match dir_type:
#         case 'train'|'val':
#             for graph in tqdm(graph_list,desc=f"Convert {dir_type} {graph_type} graph_list..."):
#                 event_stream=GraphUtils.get_event_stream(graph=graph)
#                 source_id=random.randrange(num_nodes)
#                 dataset=GraphUtils.convert_event_stream_to_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
#                 dataset_list.append(dataset)
#         case 'test':
#             for graph_id,graph in tqdm(enumerate(graph_list),desc=f"Convert {dir_type} {graph_type} graph_list..."):
#                 event_stream=GraphUtils.get_event_stream(graph=graph)
#                 for source_id in tqdm(graph.nodes,desc=f"Convert {graph_id} graph to dataset..."):
#                     dataset=GraphUtils.convert_event_stream_to_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
#                     dataset_list.append(dataset)
#     DataUtils.save_to_pickle(data=dataset_list,file_name=f"{dir_type}_{num_nodes}_{graph_type}",dir_type=dir_type,num_nodes=num_nodes)

# @staticmethod
# def compute_tR_step_old(num_nodes:int,source_id:int,edge_event:tuple=None,init:bool=False,gamma:torch.Tensor=None):
#     """
#     edge_event: tuple, (src,tar,ts)
#     gamma: [N,2] tensor, (tR,visited time)
#     """
#     if init:
#         gamma=torch.zeros((num_nodes,3),dtype=torch.float) # tR,time,ts
#         gamma[:,0]=0.0 # tR
#         gamma[:,1]=0.0 # visited time

#         gamma[source_id,0]=1.0
#         gamma[source_id,1]=0.0
#     else:
#         src,tar,ts=edge_event
#         if gamma[src,0].item()==1.0 and gamma[tar,0].item()==0.0 and gamma[src,1].item()<ts:
#             gamma[tar,0]=1.0
#             gamma[tar,1]=ts
#     return gamma

# @staticmethod
# def convert_event_stream_to_dataset_old(event_stream:list,num_nodes:int,source_id:int):
#     """
#     Input:
#         event_stream: List of edge_event tuple
#         num_nodes: number of nodes
#         source_id: source node id
#     Output:
#         dataset: sequence of data
#             data:
#                 raw: [N,1]
#                 r: [N,1]
#                 t: [N,1]
#                 src: src id
#                 tar: tar id
#                 label: tar label, 1.0 or 0.0
#                 n_mask: [N,]
#     """
#     raw=torch.zeros(num_nodes,1,dtype=torch.float32) # [N,1]
#     raw[source_id]=1.0

#     r_list=[]
#     t_list=[]
#     src_list=[]
#     tar_list=[]
#     label_list=[]
#     n_mask_list=[]

#     num_edge_events=len(event_stream)
#     neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool) # [E,N], 각 edge_event에 대한 tar의 neighbor mask
#     neighbor_history=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)] # List of [N,]
#     gamma=GraphUtils.compute_tR_step(num_nodes=num_nodes,source_id=source_id,init=True) # [N,2]
#     for i,edge_event in enumerate(event_stream):
#         # compute tR step
#         gamma=GraphUtils.compute_tR_step(num_nodes=num_nodes,source_id=source_id,edge_event=edge_event,gamma=gamma)
        
#         src,tar,ts=edge_event
#         r_list.append(gamma[:,:1]) 
#         visited_t=gamma[:,-1:]
#         t_list.append(torch.abs(visited_t-ts))
#         src_list.append(src)
#         tar_list.append(tar)
#         label_list.append(gamma[tar,0].item())

#         neighbor_history[tar][src]=True
#         neighbor_mask[i]=neighbor_history[tar] # 참조가 아닌 복사(tensor index 대입)
#         n_mask_list.append(neighbor_mask[i])
    
#     # convert to dataset, E = number of edge_events = seq_len
#     dataset={}
#     dataset['raw']=raw.unsqueeze(0).expand(num_edge_events,num_nodes,1) # [E,N,1]
#     dataset['r']=torch.stack(r_list,dim=0) # [E,N,1]
#     dataset['t']=torch.stack(t_list,dim=0) # [E,N,1]
#     dataset['src']=torch.tensor(src_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
#     dataset['tar']=torch.tensor(tar_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
#     dataset['label']=torch.tensor(label_list,dtype=torch.float32).unsqueeze(-1) # [E,1]
#     dataset['n_mask']=torch.stack(n_mask_list,dim=0) # [E,N]
#     return dataset