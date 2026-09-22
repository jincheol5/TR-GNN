import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Literal
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from graph import TemporalGraph

class TemporalGraphDataset(Dataset):
    def __init__(self,df:pd.DataFrame):
        self.src=torch.tensor(df["u"].values,dtype=torch.long)
        self.dst=torch.tensor(df["i"].values,dtype=torch.long)
        self.t=torch.tensor(df["t"].values,dtype=torch.float32)
        self.edge=torch.tensor(df["idx"].values,dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.src[idx],self.dst[idx],self.t[idx],self.edge[idx]

class EarlyStopper:
    def __init__(self,
            patience:int=1
        ):
        self.patience=patience
        self.patience_count=0
        self.best_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,
            val_loss:float,
            model:torch.nn.Module
        ):
        # val_loss가 NaN, Inf이면 즉시 early stop
        if not np.isfinite(val_loss): 
            print("Loss is NaN or Inf!")
            self.early_stop=True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return model

        # 첫 번째 validation에서는 비교할 이전 best가 없으므로 현재 loss와 모델을 그대로 best로 저장
        if self.best_state is None: 
            self.best_loss=val_loss
            self.best_state={
                key: value.detach().clone()
                for key,value in model.state_dict().items()
            }
            return model

        # val_loss가 개선 되지 않은 경우
        if self.best_loss<=val_loss: 
            self.patience_count+=1
            if self.patience<=self.patience_count:
                self.early_stop=True
                model.load_state_dict(self.best_state)
            return model

        # val_loss가 개선 된 경우
        self.patience_count=0
        self.best_loss=val_loss
        self.best_state={
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        return model

class TrainUtils:
    @staticmethod
    def set_seed(seed:int):
        os.environ["PYTHONHASHSEED"]=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    @staticmethod
    def split_graph_df(
            df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.15
        ):
        """
        Input:
            df
            train_ratio
            val_ratio
        Return:
            train_df
            val_df
            test_df
        """
        n=len(df)
        train_end=int(n*train_ratio)
        val_end=int(n*(train_ratio+val_ratio))
        train_df=df.iloc[:train_end].reset_index(drop=True)
        val_df=df.iloc[train_end:val_end].reset_index(drop=True)
        test_df=df.iloc[val_end:].reset_index(drop=True)
        return train_df,val_df,test_df

    @staticmethod
    def get_TR_result_using_Time_Centric(
            graph:TemporalGraph,
            data_loader:DataLoader
        )->dict[str,torch.Tensor]:
        """
        Input:
            graph
            data_loader
            max_hop
        Return:
            TR_result: dict
                label: [batch_seq_len,N+1,N+1], boolean tensor
                hop: [batch_seq_len,N+1,N+1], int16 tensor
                last_t: [batch_seq_len,N+1,N+1], int16 tensor
                first_t: [batch_seq_len,N+1,N+1], int16 tensor
        """
        n_node=graph.n_node
        TR_result=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        TR_hop=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_last_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_first_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        for seq_idx,(_,_,event_t,_) in enumerate(tqdm(data_loader,desc="Compute TR result tensor...")):
            query_time=event_t.max().item()
            gamma_table=graph.compute_TR_using_Time_Centric(query_time=query_time) # 모든 source-dst에 대한 TR + minimum hop 계산
            for src in range(1,n_node+1):
                for dst in range(1,n_node+1):
                    info=gamma_table[src][dst]
                    if not info["r"]:
                        continue
                    TR_result[seq_idx,src,dst]=True
                    TR_hop[seq_idx,src,dst]=int(info["hop"])
                    TR_first_t[seq_idx,src,dst]=int(info["first_t"])
                    TR_last_t[seq_idx,src,dst]=int(info["last_t"])
        return {
            "label":TR_result,
            "hop":TR_hop,
            "last_t":TR_last_t,
            "first_t":TR_first_t
        }

    @staticmethod
    def get_TR_result(
            graph:TemporalGraph,
            data_loader:DataLoader
        )->dict[str,torch.Tensor]:
        """
        Input:
            graph
            data_loader
            max_hop
        Return:
            TR_result: dict
                label: [batch_seq_len,N+1,N+1], boolean tensor
                hop: [batch_seq_len,N+1,N+1], int16 tensor
                last_t: [batch_seq_len,N+1,N+1], int16 tensor
                first_t: [batch_seq_len,N+1,N+1], int16 tensor
        """
        n_node=graph.get_num_node()
        TR_result=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        TR_hop=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_last_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_first_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        for seq_idx,(_,_,event_t,_) in enumerate(tqdm(data_loader,desc="Compute TR result tensor...")):
            query_time=event_t.max().item()
            for source in range(1,n_node+1):
                TR_info=graph.compute_TR(
                    source=source,
                    query_time=query_time
                )
                for dst in range(1,n_node+1):
                    info=TR_info[dst]
                    if not info["r"]:
                        continue

                    TR_result[seq_idx,source,dst]=True
                    TR_hop[seq_idx,source,dst]=int(info["hop"])
                    TR_first_t[seq_idx,source,dst]=int(info["first_t"])

                    # source node의 last_t는 -inf이므로 padding 값 0을 유지
                    if dst!=source:
                        TR_last_t[seq_idx,source,dst]=int(info["last_t"])
        return {
            "label":TR_result,
            "hop":TR_hop,
            "last_t":TR_last_t,
            "first_t":TR_first_t
        }