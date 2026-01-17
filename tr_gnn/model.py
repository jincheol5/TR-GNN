import torch
import torch.nn as nn
from typing_extensions import Literal
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphAttention,GraphSum
from .model_train_utils import ModelTrainUtils

class TGAT(nn.Module):
    def __init__(self,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.embedding=GraphAttention(latent_dim=latent_dim,is_memory=False)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,data_loader,device,mode=None):
        """
        Input:
            data_loader: sequence of batch
                batch:
                    init_traj: [N,1]
                    traj: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N]
                    label: [B,1]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        _,num_nodes=data_loader[0]['n_mask'].size()
        x=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            """
            embedding
            """
            embedded_emb_t=self.time_encoder(batch['emb_t']) # [B,N,latent_dim]
            z=self.embedding(traj=batch['init_traj'],x=x,delta_t_vec=embedded_emb_t,neighbor_mask=batch['n_mask'],tar_idx=batch['tar']) # [B,latent_dim]
            logit=self.linear(z)
            logit_list.append(logit)
        return logit_list

class TGN(nn.Module):
    def __init__(self,latent_dim,emb:Literal['time','sum','attn']):
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(latent_dim=latent_dim)
        match emb:
            case 'time':
                self.embedding=TimeProjection(latent_dim=latent_dim)
            case 'attn':
                self.embedding=GraphAttention(latent_dim=latent_dim)
            case 'sum':
                self.embedding=GraphSum(latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
        self.emb=emb

    def forward(self,data_loader,device,mode=None):
        """
        Input:
            data_loader: sequence of batch
                batch:
                    init_traj: [N,1]
                    traj: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N]
                    label: [B,1]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        _,num_nodes=data_loader[0]['n_mask'].size()
        pre_memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        x=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}

            """
            1. memory update
            """
            embedded_mem_t=self.time_encoder(batch['mem_t']) # [B,N,latent_dim]
            memory=self.memory_updater(traj=batch['init_traj'],memory=pre_memory,source=batch['src'],target=batch['tar'],delta_t_vec=embedded_mem_t) # [N,latent_dim]

            """
            2. embedding
            """
            embedded_emb_t=self.time_encoder(batch['emb_t']) # [B,N,latent_dim]
            match self.emb:
                case 'time':
                    z=self.embedding(memory=pre_memory,delta_t=batch['emb_t'],tar_idx=batch['tar']) # [B,latent_dim]
                case 'sum'|'attn':
                    z=self.embedding(traj=batch['init_traj'],x=x,delta_t_vec=embedded_emb_t,neighbor_mask=batch['n_mask'],tar_idx=batch['tar'],memory=pre_memory) # [B,latent_dim]
            logit=self.linear(z)
            logit_list.append(logit)
            pre_memory=memory
        return logit_list

class TR_GNN(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(latent_dim=latent_dim)
        self.embedding=GraphAttention(latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,data_loader,device,mode:Literal['train','test']='train'):
        """
        Input:
            data_loader: sequence of batch
                batch:
                    init_traj: [N,1]
                    traj: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N]
                    label: [B,1]
            device: GPU
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        _,num_nodes=data_loader[0]['n_mask'].size()
        pre_memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        x=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device) # [N,latent_dim]
        pre_traj=data_loader[0]['init_traj']
        pre_traj=pre_traj.to(device)
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}

            """
            1. memory update
            """
            embedded_mem_t=self.time_encoder(batch['mem_t']) # [B,N,latent_dim]
            memory=self.memory_updater(traj=pre_traj,memory=pre_memory,source=batch['src'],target=batch['tar'],delta_t_vec=embedded_mem_t) # [N,latent_dim]

            """
            2. embedding
            """
            embedded_emb_t=self.time_encoder(batch['emb_t']) # [B,N,latent_dim]
            z=self.embedding(traj=pre_traj,x=x,delta_t_vec=embedded_emb_t,neighbor_mask=batch['n_mask'],tar_idx=batch['tar'],memory=pre_memory) # [B,latent_dim]
            logit=self.linear(z)
            logit_list.append(logit)

            tar_pred_TR=torch.sigmoid(logit)
            tar_label_TR=batch['label']
            if mode=='train':
                tar_pred_TR=ModelTrainUtils.teacher_forcing(pred=tar_pred_TR,label=tar_label_TR)
            tar_idx=batch['tar']
            tar_idx=tar_idx.squeeze(-1)
            pre_traj[tar_idx]=tar_pred_TR
            pre_memory=memory
        return logit_list