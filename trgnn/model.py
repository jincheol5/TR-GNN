import torch
import torch.nn as nn
from typing_extensions import Literal
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphAttention,GraphSum
from .model_train_utils import ModelTrainUtils

class TGAT(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
    
    def forward(self,data_loader,device,mode:Literal['train','test']='train'):
        """
        Input:
            data_loader: List of batch
                batch
                    raw: [B,N,1]
                    t: [B,N,1]
                    tar: [B,1]
                    n_mask: [B,N,]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            raw=batch['raw'] # [B,N,1], float
            t=batch['emb_t'] # [B,N,1], float
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            """
            embedding
            """
            batch_size,num_nodes,_=raw.size()

            # target raw vector
            batch_idx=torch.arange(batch_size,device=device) # [B,]
            tar=tar.squeeze(-1) # [B,]
            tar_raw=raw[batch_idx,tar,:] # [B,1]

            # target vector
            tar_hidden_ft=torch.zeros((batch_size,self.latent_dim),dtype=torch.float,device=device) # [B,latent_dim]
            tar_vec=torch.cat([tar_raw,tar_hidden_ft],dim=-1) # [B,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=device) # [B,1]
            encoded_tar_t=self.time_encoder(tar_t) # [B,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [B,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=torch.zeros((batch_size,num_nodes,self.latent_dim),dtype=torch.float,device=device) # [B,N,latent_dim]
            x=torch.cat([raw,hidden_ft],dim=-1) # [B,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(t) # [B,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [B,N,node_dim+latent_dim+latent_dim]

            # attention result
            z=self.attention(tar_vec=tar_h,tar_idx=tar.unsqueeze(-1),h=h,neighbor_mask=n_mask) # [B,latent_dim]
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)
        return logit_list # List of [B,1], B는 seq 마다 크기 다를 수 있음

class TGN(nn.Module):
    def __init__(self,node_dim,latent_dim,emb:Literal['time','attn','sum']): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(node_dim=node_dim,latent_dim=latent_dim)
        match emb:
            case 'time':
                self.embedding=TimeProjection(latent_dim=latent_dim)
            case 'attn':
                self.embedding=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
            case 'sum':
                self.embedding=GraphSum(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
        self.emb=emb
    
    def forward(self,data_loader,device,mode:Literal['train','test']='train'):
        """
        Input:
            data_loader: List of batch
                batch
                    raw: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N,]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        batch_size,num_nodes,_=data_loader[0]['raw'].size()
        memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        mem_t=torch.zeros((batch_size,num_nodes,1),dtype=torch.float32,device=device) # [B,N,1], float
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            emb_t=batch['emb_t'] # [B,N,1], float
            raw=batch['raw'] # [B,N,1], float
            src=batch['src'] # [B,1], long
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            """
            1. update memory using previous batch raw message
            """
            delta_t=self.time_encoder(mem_t) # [B,N,latent_dim]
            updated_memory=self.memory_updater(x=raw,memory=memory,source=src,target=tar,delta_t=delta_t) # [N,latent_dim]
            mem_t=batch['mem_t'] # [B,N,1], float

            """
            2. embedding
            """
            batch_size=raw.size(0)

            # target raw vector
            batch_idx=torch.arange(batch_size,device=device) # [B,]
            tar=tar.squeeze(-1) # [B,]
            tar_raw=raw[batch_idx,tar,:] # [B,1]

            # target vec
            tar_hidden_ft=updated_memory[tar] # [B,latent_dim]
            tar_vec=torch.cat([tar_raw,tar_hidden_ft],dim=-1) # [B,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=device) # [B,1]
            encoded_tar_t=self.time_encoder(tar_t) # [B,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [B,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=updated_memory.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,latent_dim]
            x=torch.cat([raw,hidden_ft],dim=-1) # [B,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(emb_t) # [B,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [B,N,node_dim+latent_dim+latent_dim]

            # attention result
            match self.emb:
                case 'time':
                    tar_memory=updated_memory[tar] # [B,latent_dim]
                    delta_t=emb_t[batch_idx,tar,:] # [B,1]
                    z=self.embedding(target_memory=tar_memory,delta_t=delta_t) # [B,latent_dim]
                case 'attn'|'sum':
                    z=self.embedding(tar_vec=tar_h,tar_idx=tar.unsqueeze(-1),h=h,neighbor_mask=n_mask) # [B,latent_dim]
            
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)
            memory=updated_memory # set next memory
        return logit_list # List of [B,1], B는 seq 마다 크기 다를 수 있음

class TRGNN(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(node_dim=node_dim,latent_dim=latent_dim)
        self.embedding=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,data_loader,device,mode:Literal['train','test']='train'):
        """
        Input:
            data_loader: List of batch
                batch
                    raw: [B,N,1]
                    r: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N,]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        batch_size,num_nodes,_=data_loader[0]['raw'].size()
        memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        mem_t=torch.zeros((batch_size,num_nodes,1),dtype=torch.float32,device=device) # [B,N,1], float
        r=data_loader[0]['raw'][0] # [N,1]
        r=r.to(device)
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            emb_t=batch['emb_t'] # [B,N,1], float
            
            src=batch['src'] # [B,1], long
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            batch_size=src.size(0)
            r=r.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]

            """
            1. update memory
            """
            delta_t=self.time_encoder(mem_t) # [B,N,latent_dim]
            updated_memory=self.memory_updater(x=r,memory=memory,source=src,target=tar,delta_t=delta_t) # [N,latent_dim]
            mem_t=batch['mem_t'] # [B,N,1], float

            """
            2. embedding
            """
            # target r vector
            batch_idx=torch.arange(batch_size,device=device) # [B,]
            tar=tar.squeeze(-1) # [B,]
            tar_r=r[batch_idx,tar,:] # [B,1]

            # target vec
            tar_hidden_ft=updated_memory[tar] # [B,latent_dim]
            tar_vec=torch.cat([tar_r,tar_hidden_ft],dim=-1) # [B,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=device) # [B,1]
            encoded_tar_t=self.time_encoder(tar_t) # [B,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [B,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=updated_memory.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,latent_dim]
            x=torch.cat([r,hidden_ft],dim=-1) # [B,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(emb_t) # [B,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [B,N,node_dim+latent_dim+latent_dim]

            # attention result
            z=self.embedding(tar_vec=tar_h,tar_idx=tar.unsqueeze(-1),h=h,neighbor_mask=n_mask) # [B,latent_dim]

            # compute logit and set next memory 
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)
            memory=updated_memory # set next memory

            # set next r
            r_pred=r[-1].clone() # [N,1]
            pred_logits=torch.sigmoid(logit)
            pred_logits.squeeze(1) # [B,]
            r_pred[tar]=pred_logits
            r_label=batch['r'][-1].clone() # [N,1]
            if mode=='train':
                r=ModelTrainUtils.teacher_forcing(r_pred=r_pred,r_label=r_label,tar=tar) # [N,1]
            else:
                r=r_pred
        return logit_list # List of [B,1], B는 seq 마다 크기 다를 수 있음

class TRGAT(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.embedding=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
    
    def forward(self,data_loader,device,mode:Literal['train','test']='train'):
        """
        Input:
            data_loader: List of batch
                batch
                    raw: [B,N,1]
                    emb_t: [B,N,1]
                    tar: [B,1]
                    n_mask: [B,N,]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        num_nodes=data_loader[0]['raw'].size(1)
        memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        r=data_loader[0]['raw'][0] # [N,1]
        r=r.to(device)
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            emb_t=batch['emb_t'] # [B,N,1], float
            src=batch['src'] # [B,1], long
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            batch_size=src.size(0)
            r=r.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]

            """
            embedding
            """
            # target r vector
            batch_idx=torch.arange(batch_size,device=device) # [B,]
            tar=tar.squeeze(-1) # [B,]
            tar_r=r[batch_idx,tar,:] # [B,1]

            # target vec
            tar_hidden_ft=memory[tar] # [B,latent_dim]
            tar_vec=torch.cat([tar_r,tar_hidden_ft],dim=-1) # [B,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=device) # [B,1]
            encoded_tar_t=self.time_encoder(tar_t) # [B,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [B,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=memory.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,latent_dim]
            x=torch.cat([r,hidden_ft],dim=-1) # [B,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(emb_t) # [B,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [B,N,node_dim+latent_dim+latent_dim]

            # attention result
            z=self.embedding(tar_vec=tar_h,tar_idx=tar.unsqueeze(-1),h=h,neighbor_mask=n_mask) # [B,latent_dim]

            # compute logit and set next r
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)

            # set next memory
            memory[tar]=z  

            # set next r
            r_pred=r[-1].clone() # [N,1]
            pred_logits=torch.sigmoid(logit)
            pred_logits.squeeze(1) # [B,]
            r_pred[tar]=pred_logits
            r_label=batch['r'][-1].clone() # [N,1]
            if mode=='train':
                r=ModelTrainUtils.teacher_forcing(r_pred=r_pred,r_label=r_label,tar=tar) # [N,1]
            else:
                r=r_pred
        return logit_list # List of [B,1], B는 seq 마다 크기 다를 수 있음