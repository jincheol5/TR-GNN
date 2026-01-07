import torch
import torch.nn as nn
from typing_extensions import Literal
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphSum,GraphAttention

class TGAT(nn.Module):
    def __init__(self,traj_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=traj_dim+latent_dim,latent_dim=latent_dim,is_memory=False) # node_dim=traj_dim+latent_dim
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.traj_dim=traj_dim
        self.latent_dim=latent_dim

    def forward(self,batch,device):
        """
        Input:
            batch: dict
                init_traj: [N,1]
                traj: [B,N,1]
                emb_t: [B,N,1]
                mem_t: [B,N,1]
                src: [B,1]
                tar: [B,1]
                n_mask: [B,N]
                label: [B,1]
        Output:
            logit: [B,1], B는 seq 마다 크기 다를 수 있음
        """
        batch_size,num_nodes,_=batch['traj'].size()
        raw=torch.zeros((batch_size,num_nodes,self.latent_dim),device=device) # [B,N,latent_dim], node raw feature
        init_traj=batch['init_traj'] # [N,1]
        init_traj=init_traj.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]
        emb_t=batch['emb_t'] # [B,N,1]
        tar=batch['tar'] # [B,1]
        n_mask=batch['n_mask'] # [B,N]
        x=torch.cat([init_traj,raw],dim=-1) # [B,N,node_dim], node_dim=1+latent_dim 
        delta_t_vec=self.time_encoder(emb_t) # [B,N,latent_dim]

        """
        Embedding
        """
        z=self.attention(x=x,delta_t_vec=delta_t_vec,neighbor_mask=n_mask,tar_idx=tar) # [B,latent_dim]
        logit=self.linear(z) # [B,1]
        return logit # [B,1], B는 seq 마다 크기 다를 수 있음

class TGN(nn.Module):
    def __init__(self,traj_dim,latent_dim,emb:Literal['time','sum','attn']): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(latent_dim=latent_dim)
        match emb:
            case 'time':
                self.embedding=TimeProjection(latent_dim=latent_dim)
            case 'sum':
                self.embedding=GraphSum(node_dim=traj_dim+latent_dim,latent_dim=latent_dim)
            case 'attn':
                self.embedding=GraphAttention(node_dim=traj_dim+latent_dim,latent_dim=latent_dim,is_memory=True)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
        self.emb=emb

    def forward(self,batch,pre_memory=None,device=None):
        """
        Input:
            batch: dict
                init_traj: [N,1]
                traj: [B,N,1]
                emb_t: [B,N,1]
                mem_t: [B,N,1]
                src: [B,1]
                tar: [B,1]
                n_mask: [B,N]
                label: [B,1]
            pre_memory: [N,1]
        Output:
            logit: [B,1], B는 seq 마다 크기 다를 수 있음
        """
        batch_size,num_nodes,_=batch['traj'].size()
        raw=torch.zeros((batch_size,num_nodes,self.latent_dim),device=device) # [B,N,latent_dim], node raw feature
        init_traj=batch['init_traj'] # [N,1]
        init_traj=init_traj.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]
        x=torch.cat([init_traj,raw],dim=-1) # [B,N,node_dim], node_dim=1+latent_dim 
        mem_t=batch['mem_t'] # [B,N,1]
        emb_t=batch['emb_t'] # [B,N,1]
        src=batch['src'] # [B,1]
        tar=batch['tar'] # [B,1]
        n_mask=batch['n_mask'] # [B,N]
        
        """
        1. memory update using previous raw messages
        """
        if pre_memory==None:
            pre_memory=torch.zeros((num_nodes,self.latent_dim),device=device) # [N,latent_dim]
        delta_mem_t_vec=self.time_encoder(mem_t) # [B,N,latent_dim]
        updated_memory=self.memory_updater(memory=pre_memory,source=src,target=tar,delta_t_vec=delta_mem_t_vec) # [N,latent_dim]
        updated_memory=updated_memory.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,latent_dim]

        """
        2. embedding
        """
        delta_emb_t_vec=self.time_encoder(emb_t) # [B,N,latent_dim]
        match self.emb:
            case 'time':
                # TimeProjection expects memory: [N,latent_dim], delta_t: [B,1], tar_idx: [B,1]
                # updated_memory is [B,N,latent_dim] (expanded) -> use the unbatched memory (first element)
                batch_idx=torch.arange(batch_size,device=device)
                tar_idx_s=tar.squeeze(-1) # [B,]
                delta_t_target=emb_t[batch_idx,tar_idx_s] # [B,1]
                z=self.embedding(memory=updated_memory[0],delta_t=delta_t_target,tar_idx=tar) # [B,latent_dim]
            case 'sum':
                z=self.embedding(x=x,delta_t_vec=delta_emb_t_vec,neighbor_mask=n_mask,tar_idx=tar,memory=updated_memory) # [B,latent_dim]
            case 'attn':
                z=self.embedding(x=x,delta_t_vec=delta_emb_t_vec,neighbor_mask=n_mask,tar_idx=tar,memory=updated_memory) # [B,latent_dim]
        logit=self.linear(z) # [B,1]
        updated_memory=updated_memory[0] # [N,latent_dim]
        return logit,updated_memory # [B,1], B는 seq 마다 크기 다를 수 있음