import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal
from torch_scatter import scatter_mean

class TimeEncoder(nn.Module):
    def __init__(self,time_dim:int,parameter_requires_grad:bool=True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder,self).__init__()
        self.time_dim=time_dim
        self.w=nn.Linear(1,time_dim)
        self.w.weight=nn.Parameter((torch.from_numpy(1/10**np.linspace(0,9,time_dim,dtype=np.float32))).reshape(time_dim,-1))
        self.w.bias=nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad=False
            self.w.bias.requires_grad=False

    def forward(self,timestamps:torch.Tensor):
        """
        compute time encodings of time in timestamps
        Input:
            timestamps: [B,N,1]
        Output:
            updated_timestamps: [B,N,time_dim]
        """
        output=torch.cos(self.w(timestamps)) # [B,N,time_dim]
        return output

class MemoryUpdater(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.src_mlp=nn.Sequential(
            nn.Linear(in_features=latent_dim+latent_dim+latent_dim,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim)
        )
        self.tar_mlp=nn.Sequential(
            nn.Linear(in_features=latent_dim+latent_dim+latent_dim,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim)
        )
        self.gru=nn.GRUCell(input_size=latent_dim,hidden_size=latent_dim)
    
    def message_aggregate(self,source:torch.Tensor,target:torch.Tensor,source_msg:torch.Tensor,target_msg:torch.Tensor):
        """
        Input:
            source: [B,1]
            target: [B,1]
            source_msg: [B,latent_dim]
            target_msg: [B,latent_dim]
        Output:
            aggregated_msg: [unique_node_size,latent_dim]
        """
        src_tar_nodes=torch.cat([source,target],dim=0).squeeze(-1) # [2*B,]
        src_tar_msg=torch.cat([source_msg,target_msg],dim=0)  # [2*B,latent_dim]
        unique_nodes,inverse_indices=torch.unique(src_tar_nodes,return_inverse=True) # [unique_node_size,],[2*B]
        aggregated_msg=scatter_mean(src=src_tar_msg,index=inverse_indices,dim=0)  # [unique_node_size,latent_dim]
        return unique_nodes,aggregated_msg # [unique_node_size,],[unique_node_size,latent_dim]

    def forward(self,memory,source:torch.Tensor,target:torch.Tensor,delta_t_vec:torch.Tensor):
        """
        Input:
            memory: [N,latent_dim]
            source: [B,1]
            target: [B,1]
            delta_t_vec: [B,N,latent_dim]
        Output:
            updated_memory
        """
        batch_size=source.size(0)

        source_batch_indices=torch.arange(batch_size,device=memory.device) # [B,]
        source=source.squeeze(-1) # [B,]
        source_memory=memory[source] # [B,latent_dim]
        source_delta_t_vec=delta_t_vec[source_batch_indices,source,:] # [B,latent_dim]
        
        target_batch_indices=torch.arange(batch_size,device=memory.device)
        target=target.squeeze(-1) # [B,]
        target_memory=memory[target] # [B,latent_dim]
        target_delta_t_vec=delta_t_vec[target_batch_indices,target,:] # [B,latent_dim]

        source_msg_input=torch.cat([source_memory,target_memory,source_delta_t_vec],dim=-1) # [B,latent_dim+latent_dim+latent_dim]
        source_msg=self.src_mlp(source_msg_input) # [B,latent_dim]

        target_msg_input=torch.cat([target_memory,source_memory,target_delta_t_vec],dim=-1) # [B,latent_dim+latent_dim+latent_dim]
        target_msg=self.tar_mlp(target_msg_input) # [B,latent_dim]

        unique_nodes,aggregated_msg=self.message_aggregate(
            source=source.unsqueeze(-1),
            target=target.unsqueeze(-1),
            source_msg=source_msg,
            target_msg=target_msg
        ) # [unique_node_size,],[unique_node_size,latent_dim]

        pre_memory=memory[unique_nodes] # [unique_node_size,latent_dim]
        new_memory=self.gru(aggregated_msg,pre_memory) # [unique_node_size,latent_dim]
        memory[unique_nodes]=new_memory # [N,latent_dim]
        return memory # [N,latent_dim]

"""
Embedding Modules
1. TimeProjection
2. GraphSum
3. GraphAttention
"""
class TimeProjection(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.w=nn.Linear(in_features=1,out_features=latent_dim)

    def forward(self,memory,delta_t,tar_idx):
        """
        Input:
            memory: [N,latent_dim] 
            delta_t: [B,1]
            tar_idx: [B,1]
        Output:
            z: [B,latent_dim]
        """
        tar_memory=memory[tar_idx.squeeze(-1)] # [B,latent_dim]
        delta_t_vec=self.w(delta_t) # [B,latent_dim]
        delta_t_vec=delta_t_vec+1 # [B,latent_dim]
        z=torch.mul(delta_t_vec,tar_memory) # [B,latent_dim]
        return z

class GraphSum(nn.Module):
    def __init__(self,node_dim,latent_dim):
        super().__init__()
        self.w_1=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.w_2=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.relu=nn.ReLU()

    def forward(self,x,delta_t_vec,neighbor_mask,tar_idx,memory):
        """
        Input:
            x: [B,N,node_dim]
            delta_t_vec: [B,N,latent_dim]
            neighbor_mask: [B,N]
            tar_idx: [B,1]
            memory: [B,N,latent_dim]
        Output:
            z: [B,latent_dim]
        """
        batch_size=x.size(0)
        batch_idx=torch.arange(batch_size,device=x.device)
        tar_idx=tar_idx.squeeze(-1)  # [B,]

        # 이웃 노드 하나도 없는 경우 확인->없을 경우 자기 자신만 true가 되도록 mask 수정
        no_neighbor=~neighbor_mask.any(dim=1)  # [B,] bool vec, 이웃 없는 행은 true로
        if no_neighbor.any():
            neighbor_mask[no_neighbor,tar_idx[no_neighbor]] = True

        # compute node-wise projection: [B,N,latent_dim]
        w_1_input=torch.cat([x,memory,delta_t_vec], dim=-1)  # [B,N,node_dim+latent_dim+latent_dim]
        w_1_output=self.w_1(w_1_input)  # [B,N,latent_dim]

        # aggregate neighbor messages per batch: [B,latent_dim]
        neighbor_mask_f=neighbor_mask.float().unsqueeze(-1)  # [B,N,1]
        w_1_output_sum=(neighbor_mask_f*w_1_output).sum(dim=1)  # [B,latent_dim]
        h_hat=self.relu(w_1_output_sum)  # [B,latent_dim]

        # target node features per batch
        tar_x=x[batch_idx,tar_idx]  # [B,node_dim]
        tar_memory=memory[batch_idx,tar_idx]  # [B,latent_dim]
        w_2_input=torch.cat([tar_x,tar_memory,h_hat], dim=-1)  # [B,node_dim+latent_dim+latent_dim]
        z = self.w_2(w_2_input)  # [B,latent_dim]
        return z

class GraphAttention(nn.Module):
    def __init__(self,node_dim,latent_dim,is_memory:bool=True):
        super().__init__()
        if is_memory:
            self.query_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
            self.key_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
            self.value_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
            self.ffn=nn.Sequential(
                nn.Linear(latent_dim+node_dim+latent_dim+latent_dim,latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim,latent_dim)
            )
        else:
            self.query_linear=nn.Linear(in_features=node_dim+latent_dim,out_features=latent_dim)
            self.key_linear=nn.Linear(in_features=node_dim+latent_dim,out_features=latent_dim)
            self.value_linear=nn.Linear(in_features=node_dim+latent_dim,out_features=latent_dim)
            self.ffn=nn.Sequential(
                nn.Linear(latent_dim+node_dim+latent_dim,latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim,latent_dim)
            )
        self.is_memory=is_memory

    def forward(self,x,delta_t_vec,neighbor_mask,tar_idx,memory=None):
        """
        Input:
            x: [B,N,node_dim], src_info||raw_feature of node 
            delta_t_vec: [B,N,latent_dim]
            neighbor_mask: [B,N,], neighbor node mask
            tar_idx: [B,1], long
            memory: [B,N,latent_dim] or None
        Output:
            z: [B,latent_dim]
        """
        batch_size=x.size(0)
        batch_idx=torch.arange(batch_size,device=x.device) # [B,]
        tar_idx=tar_idx.squeeze(-1) # [B,]

        if self.is_memory:
            tar_x=x[batch_idx,tar_idx] # [B,node_dim]
            tar_memory=memory[batch_idx,tar_idx] # [B,latent_dim]
            q_input=torch.cat([tar_x,delta_t_vec[batch_idx,tar_idx],tar_memory],dim=-1) # [B,node_dim+latent_dim+latent_dim]
            kv_input=torch.cat([x,delta_t_vec,memory],dim=-1) # [B,N,node_dim+latent_dim+latent_dim] 
        else:
            tar_x=x[batch_idx,tar_idx] # [B,node_dim]
            q_input=torch.cat([tar_x,delta_t_vec[batch_idx,tar_idx]],dim=-1) # [B,node_dim+latent_dim]
            kv_input=torch.cat([x,delta_t_vec],dim=-1) # [B,N,node_dim+latent_dim]

        q=self.query_linear(q_input) # [B,latent_dim]
        k=self.key_linear(kv_input) # [B,N,latent_dim]
        v=self.value_linear(kv_input) # [B,N,latent_dim]

        q=q.unsqueeze(1) # [B,1,latent_dim]

        feature_dim=q.size(-1)
        attention_scores=torch.matmul(q,k.transpose(1,2))/(feature_dim**0.5) # [B,1,N]

        # 이웃 노드 하나도 없는 경우 확인->없을 경우 자기 자신만 true가 되도록 mask 수정
        no_neighbor=~neighbor_mask.any(dim=1) # [B,] bool vec, 이웃 없는 행은 true로
        if no_neighbor.any():
            neighbor_mask[no_neighbor,tar_idx[no_neighbor,0]]=True

        neighbor_mask=neighbor_mask.unsqueeze(1) # [B,1,N]
        attention_scores=attention_scores.masked_fill(~neighbor_mask,float('-inf')) # [B,1,N]

        attention_weight=F.softmax(attention_scores,dim=-1) # [B,1,N]

        neighbor_weight_sum=torch.matmul(attention_weight,v) # [B,1,latent_dim]
        neighbor_weight_sum=neighbor_weight_sum.squeeze(1) # [B,latent_dim]

        z=torch.cat([neighbor_weight_sum,q_input],dim=-1) 
        z=self.ffn(z) # [B,latent_dim]
        return z