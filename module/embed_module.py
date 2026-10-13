import torch
import torch.nn as nn
from graph import TGN_Graph
from .time_encoder_module import TimeEncoder
from .attn_module import TemporalGraphAttn

class GraphEmbeddingModule(nn.Module):
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            time_dim:int=32,
            latent_dim:int=32, 
            mem_dim:int=32,
            embed_dim:int=32,
            n_layer:int=1,
            n_neighbor:int=5,
            use_last_state:bool=False,
            use_memory:bool=False,
            time_encoder:TimeEncoder=None
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.embed_dim=embed_dim
        self.time_dim=time_dim
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.use_last_state=use_last_state
        self.use_memory=use_memory
        if use_memory:
            self.mem_dim=mem_dim

        # module
        self.time_encoder=time_encoder

    def set_graph(self,graph:TGN_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def compute_embedding(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            last_state:torch.Tensor|None=None,
            mem_vec:torch.Tensor|None=None,
            n_layer:int=1
        ):
        """
        Input:
            tar: [n_tar,]
            tar_t: [n_tar,]
            last_state: [N+1,]
            mem_vec: [N+1,mem_dim]
            n_layer: int
        Return:
            updated_tar_ft: [n_tar,output_dim]
        """
        ### set tar_ft
        if self.use_last_state and last_state is not None:
            tar_ft=self.graph.get_node_ft(node=tar) # [n_tar,node_dim]
            tar_last_state=last_state[tar].unsqueeze(-1) # [n_tar,1]
            tar_ft=torch.cat([tar_last_state,tar_ft],dim=-1) # [n_tar,1+node_dim]
        else:
            tar_ft=self.graph.get_node_ft(node=tar) # [n_tar,node_dim]

        ### set tar_mem
        if self.use_memory and mem_vec is not None:
            tar_mem=mem_vec[tar]
            tar_ft=torch.concat(
                [tar_ft,tar_mem],
                dim=-1
            ) # [n_tar,node_dim+mem_dim] or [n_tar,1+node_dim+mem_dim]

        if n_layer==0:
            return tar_ft
        else:
            tar_vec=self.compute_embedding(
                tar=tar,
                tar_t=tar_t,
                last_state=last_state,
                mem_vec=mem_vec,
                n_layer=n_layer-1
            ) # [n_tar,tar_dim] if n_layer=1 else # [n_tar,output_dim]

            temporal_neighbor=self.graph.get_temporal_neighbor(
                tar=tar,
                tar_t=tar_t,
                n_neighbor=self.n_neighbor
            )
            neighbor=temporal_neighbor["neighbor"]
            neighbor_mask=temporal_neighbor["neighbor_mask"]
            neighbor_t=temporal_neighbor["neighbor_t"]
            neighbor_ts=temporal_neighbor["neighbor_ts"]
            neighbor_edge=temporal_neighbor["neighbor_edge"]

            # flatten for neighbor embedding
            n_tar,n_neighbor=neighbor.size()
            neighbor=neighbor.flatten() # [n_tar,n_neighbor] -> [n_tar x n_neighbor,]
            neighbor_t=neighbor_t.flatten() # [n_tar,n_neighbor] -> [n_tar x n_neighbor,]
            
            # apply time encoding
            tar_ts=torch.zeros_like(tar,dtype=torch.float32,device=tar.device).unsqueeze(-1) # [n_tar,1]
            tar_ts_vec=self.time_encoder(tar_ts) # [n_tar,time_dim]
            neighbor_ts=neighbor_ts.unsqueeze(-1) # [n_tar,n_neighbor,1]
            neighbor_ts_vec=self.time_encoder(neighbor_ts) # [n_tar,n_neighbor,time_dim]

            # get neighbor embedding
            neighbor_vec=self.compute_embedding(
                tar=neighbor,
                tar_t=neighbor_t,
                last_state=last_state,
                mem_vec=mem_vec,
                n_layer=n_layer-1
            ) # [n_tar x n_neighbor,tar_dim] if n_layer=1 else [n_tar x n_neighbor,output_dim] 

            # reshape
            neighbor_vec=neighbor_vec.reshape(n_tar,n_neighbor,-1) # -> [n_tar,n_neighbor,tar_dim] or [n_tar,n_neighbor,output_dim]

            # get edge feature
            neighbor_edge_ft=self.graph.get_edge_ft(edge=neighbor_edge) # [n_tar,n_neighbor,edge_dim]

            ### Aggregation
            updated_tar_vec=self.aggregate(
                tar_vec=tar_vec,
                tar_ts_vec=tar_ts_vec,
                neighbor_vec=neighbor_vec,
                neighbor_ts_vec=neighbor_ts_vec,
                neighbor_edge_ft=neighbor_edge_ft,
                neighbor_mask=neighbor_mask,
                n_layer=n_layer
            )
            return updated_tar_vec

    def aggregate(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor,
            n_layer:int
        ):
        return NotImplemented

class GraphAttnEmbedding(GraphEmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            time_dim:int=32,
            latent_dim:int=32,
            mem_dim:int=32, 
            embed_dim:int=32,
            n_layer:int=1,
            n_neighbor:int=5,
            n_head:int=1,
            use_last_state:bool=False,
            use_memory:bool=False,
            time_encoder:TimeEncoder=None
        ):
        super(GraphAttnEmbedding,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            latent_dim=latent_dim,
            mem_dim=mem_dim,
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            use_last_state=use_last_state,
            use_memory=use_memory,
            time_encoder=time_encoder
        )
        # module
        layer_0_input_dim=(1 if self.use_last_state else 0)+node_dim+(mem_dim if self.use_memory else 0)
        self.attn_layers=torch.nn.ModuleList(
            [
                TemporalGraphAttn(
                    input_dim=layer_0_input_dim if idx==0 else embed_dim,
                    edge_dim=edge_dim,
                    latent_dim=latent_dim,
                    embed_dim=embed_dim,
                    time_dim=time_dim,
                    n_head=n_head
                )
                for idx in range(n_layer)
            ]
        )

    def aggregate(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor,
            n_layer:int
        ):
        """
        Input:
            tar_vec: [n_tar,tar_dim] or [n_tar,output_dim]
            tar_ts_vec: [n_tar,time_dim]
            neighbor_vec: [n_tar,n_neighbor,tar_dim] or [n_tar,n_neighbor,output_dim]
            neighbor_ts_vec: [n_tar,n_neighbor,time_dim]
            neighbor_edge_ft: [n_tar,n_neighbor,edge_dim]
            neighbor_mask: [n_tar,n_neighbor]
            n_layer: int
        """
        aggr_module=self.attn_layers[n_layer-1]
        output=aggr_module(
            tar_vec=tar_vec,
            tar_ts_vec=tar_ts_vec,
            neighbor_vec=neighbor_vec,
            neighbor_ts_vec=neighbor_ts_vec,
            neighbor_edge_ft=neighbor_edge_ft,
            neighbor_mask=neighbor_mask
        ) # [n_tar,output_dim]
        return output 
