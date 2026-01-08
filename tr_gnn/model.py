import torch
import torch.nn as nn
from typing_extensions import Literal
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphSum,GraphAttention
from .model_train_utils import ModelTrainUtils

class TGAT(nn.Module):
    def __init__(self,traj_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=traj_dim+latent_dim,latent_dim=latent_dim,is_memory=False) # node_dim=traj_dim+latent_dim
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.traj_dim=traj_dim
        self.latent_dim=latent_dim

    def forward(self,data_loader,device=None,mode=None):
        """
        Input:
            data_loader: list of batch
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
            logit_list: list of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        init_traj=data_loader[0]['init_traj'] # [N,1]
        init_traj=init_traj.to(device)
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            batch_size,num_nodes,_=batch['traj'].size()
            expanded_init_traj=init_traj.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]
            raw=torch.zeros((batch_size,num_nodes,self.latent_dim),device=device) # [B,N,latent_dim], node raw feature
            x=torch.cat([expanded_init_traj,raw],dim=-1) # [B,N,node_dim], node_dim=1+latent_dim 

            emb_t=batch['emb_t'] # [B,N,1]
            tar=batch['tar'] # [B,1]
            n_mask=batch['n_mask'] # [B,N]
            delta_t_vec=self.time_encoder(emb_t) # [B,N,latent_dim]

            """
            Embedding
            """
            z=self.attention(x=x,delta_t_vec=delta_t_vec,neighbor_mask=n_mask,tar_idx=tar) # [B,latent_dim]
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)
        return logit_list # list of [B,1], B는 seq 마다 크기 다를 수 있음

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

    def forward(self,data_loader,device=None,mode=None):
        """
        Input:
            data_loader: list of batch
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
            logit_list: list of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        num_nodes=data_loader[0]['traj'].size(1)
        init_traj=data_loader[0]['init_traj'] # [N,1]
        init_traj=init_traj.to(device)
        memory=torch.zeros((num_nodes,self.latent_dim),device=device) # [N,latent_dim]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            batch_size,num_nodes,_=batch['traj'].size()
            expanded_init_traj=init_traj.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]
            raw=torch.zeros((batch_size,num_nodes,self.latent_dim),device=device) # [B,N,latent_dim], node raw feature
            x=torch.cat([expanded_init_traj,raw],dim=-1) # [B,N,node_dim], node_dim=1+latent_dim 

            mem_t=batch['mem_t'] # [B,N,1]
            emb_t=batch['emb_t'] # [B,N,1]
            src=batch['src'] # [B,1]
            tar=batch['tar'] # [B,1]
            n_mask=batch['n_mask'] # [B,N]
        
            """
            1. memory update using previous raw messages
            """
            delta_mem_t_vec=self.time_encoder(mem_t) # [B,N,latent_dim]
            updated_memory=self.memory_updater(memory=memory,source=src,target=tar,delta_t_vec=delta_mem_t_vec) # [N,latent_dim]
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
            logit_list.append(logit)
            memory=updated_memory[0] # [N,latent_dim]
        return logit_list # list of [B,1], B는 seq 마다 크기 다를 수 있음

class TR_GNN(nn.Module):
    def __init__(self,traj_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(latent_dim=latent_dim)
        self.embedding=GraphAttention(node_dim=traj_dim+latent_dim,latent_dim=latent_dim,is_memory=True)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
    
    def forward(self,data_loader,device=None,mode=None):
        """
        Input:
            data_loader: list of batch
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
            logit_list: list of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        num_nodes=data_loader[0]['traj'].size(1)
        pre_traj=data_loader[0]['init_traj'] # [N,1]
        pre_traj=pre_traj.to(device)
        memory=torch.zeros((num_nodes,self.latent_dim),device=device) # [N,latent_dim]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            batch_size,num_nodes,_=batch['traj'].size()
            expanded_pre_traj=pre_traj.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]
            raw=torch.zeros((batch_size,num_nodes,self.latent_dim),device=device) # [B,N,latent_dim], node raw feature
            x=torch.cat([expanded_pre_traj,raw],dim=-1) # [B,N,node_dim], node_dim=1+latent_dim 

            mem_t=batch['mem_t'] # [B,N,1]
            emb_t=batch['emb_t'] # [B,N,1]
            src=batch['src'] # [B,1]
            tar=batch['tar'] # [B,1]
            n_mask=batch['n_mask'] # [B,N]
        
            """
            1. memory update using previous raw messages
            """
            delta_mem_t_vec=self.time_encoder(mem_t) # [B,N,latent_dim]
            updated_memory=self.memory_updater(memory=memory,source=src,target=tar,delta_t_vec=delta_mem_t_vec) # [N,latent_dim]
            updated_memory=updated_memory.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,latent_dim]

            """
            2. embedding
            """
            delta_emb_t_vec=self.time_encoder(emb_t) # [B,N,latent_dim]
            z=self.embedding(x=x,delta_t_vec=delta_emb_t_vec,neighbor_mask=n_mask,tar_idx=tar,memory=updated_memory) # [B,latent_dim]
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)

            pred_logit=torch.sigmoid(logit) # [B,1]
            tar_label=batch['label'] # [B,1]
            if mode=="train":
                pred_logit=ModelTrainUtils.teacher_forcing(pred=pred_logit,label=tar_label)
            tar=tar.squeeze(1) # [B,]
            pre_traj[tar]=pred_logit # [N,1]
            memory=updated_memory[0] # [N,latent_dim]
        return logit_list # list of [B,1], B는 seq 마다 크기 다를 수 있음