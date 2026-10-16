import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Literal
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,Subset
from graph import TemporalGraph,TGN_Graph,DyGFormer_Graph
from .graph_utils import GraphAlgorithm
from .sampling_utils import SamplingUtils

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
    def split_data_loader(
            data_loader:DataLoader,
            train_ratio:float=0.7,
            val_ratio:float=0.15
        ):
        """
        len(data_loader)의 배치 수를 기준으로 train/val/test를 분할.
        train과 val 배치 수는 각각 비율을 곱해 내림하고 test는 나머지 샘플 전부를 사용. 
        batch_size를 유지하며 마지막 배치도 포함.
        입력 sampler의 순서를 한 번 확정한 뒤 각 loader는 shuffle 없이 사용.
        """
        n_batch=len(data_loader)
        batch_size=data_loader.batch_size
        indices=list(data_loader.sampler)
        train_end=min(int(n_batch*train_ratio)*batch_size,len(indices))
        val_end=min(train_end+int(n_batch*val_ratio)*batch_size,len(indices))
        loader_kwargs={
            "batch_size":batch_size,
            "shuffle":False,
            "drop_last":False,
            "collate_fn":data_loader.collate_fn,
            "num_workers":data_loader.num_workers,
            "pin_memory":data_loader.pin_memory,
            "worker_init_fn":data_loader.worker_init_fn,
            "generator":data_loader.generator,
        }
        if data_loader.num_workers>0:
            loader_kwargs.update(
                prefetch_factor=data_loader.prefetch_factor,
                persistent_workers=data_loader.persistent_workers,
                multiprocessing_context=data_loader.multiprocessing_context,
                timeout=data_loader.timeout
            )
        train_loader=DataLoader(Subset(data_loader.dataset,indices[:train_end]),**loader_kwargs)
        val_loader=DataLoader(Subset(data_loader.dataset,indices[train_end:val_end]),**loader_kwargs)
        test_loader=DataLoader(Subset(data_loader.dataset,indices[val_end:]),**loader_kwargs)
        return train_loader,val_loader,test_loader

    @staticmethod
    def convert_batch_to_eventstream(
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        )->list[tuple[int,int,float]]:
        """
        (src, dst, event_t) 배치를 (src, dst, t) 튜플 리스트로 변환.
        """
        eventstream=[]
        eventstream.extend(
            (int(source),int(destination),float(timestamp))
            for source,destination,timestamp in zip(
                src.tolist(),dst.tolist(),event_t.tolist()
            )
        )
        return eventstream

    @staticmethod
    def get_TR_result_using_Time_Centric(
            graph:TemporalGraph,
            data_loader:DataLoader
        )->dict[str,torch.Tensor]:
        """
        Input:
            graph
            data_loader
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
        gamma_table=None
        for seq_idx,(src,dst,event_t,_) in enumerate(tqdm(data_loader,desc="Compute TR result tensor...")):
            eventstream=TrainUtils.convert_batch_to_eventstream(src=src,dst=dst,event_t=event_t)
            gamma_table=GraphAlgorithm.compute_TR_using_Time_Centric(
                eventstream=eventstream,
                n_node=n_node,
                gamma_table=gamma_table
            )
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
    def get_TR_sample_loader(
            source:int,
            data_loader:DataLoader,
            TR_result:dict[str,torch.Tensor]|None=None,
            sampling:Literal["random","focused","evaluate"]=f"focused",
            n_sample:int=200,
            n_pair:int=10
        )->list[dict[str,torch.Tensor]]:
        """
        Input:
            data_loader
            n_pair
            TR_result
            sampling
        Return:
            TR_sample_loader
        """
        TR_label=TR_result["label"]
        TR_hop=TR_result["hop"]
        TR_sample_loader=[]
        for batch_idx,(_,dst,event_t,_) in tqdm(
                enumerate(data_loader),
                desc="Generating TR samples..."
            ):
            query_time=event_t.max().item()
            match sampling:
                case "random":
                    TR_sample=SamplingUtils.source_random_TR_sampling(
                        n_sample=n_sample,
                        n_pair=n_pair,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx]
                    )
                case "focused":
                    TR_sample=SamplingUtils.source_focused_TR_sampling(
                        source=source,
                        dst=dst,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx]
                    )
                case "evaluate":
                    TR_sample=SamplingUtils.evaluate_TR_sampling(
                        n_sample=200,
                        source=source,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx],
                        TR_hop=TR_hop[batch_idx]
                    )
            TR_sample_loader.append(TR_sample)
        return TR_sample_loader

    @staticmethod
    def get_graph_list(
            graph_df_list:list[pd.DataFrame],
            model_name:Literal["TGAT","TGN","DyGFormer","ReaCH-TGN","TR-GNN"],
            node_dim:int,
            edge_dim:int
        ):
        """
        convert graph_df_list to graph_list
        """
        graph_list=[]
        for graph_df in graph_df_list:
            if model_name=="DyGFormer":
                graph=DyGFormer_Graph(
                    graph_df=graph_df,
                    node_dim=node_dim,
                    edge_dim=edge_dim
                )
            else: # TGAT, TGN, ReaCH-TGN, TR-GNN
                graph=TGN_Graph(
                    graph_df=graph_df,
                    node_dim=node_dim,
                    edge_dim=edge_dim
                )
            graph_list.append(graph)
        return graph_list

    @staticmethod
    def get_graph_loader_list(
            graph_df_list:list[pd.DataFrame],
            batch_size:int
        ):
        """
        convert graph_df_list to graph_loader_list
        """
        graph_loader_list=[]
        for graph_df in graph_df_list:
            graph_dataset=TemporalGraphDataset(df=graph_df)
            graph_loader=DataLoader(graph_dataset,batch_size=batch_size,shuffle=False)
            graph_loader_list.append(graph_loader)
        return graph_loader_list

    @staticmethod
    def get_sample_loader_list(
            n_node:int,
            graph_loader_list:list[DataLoader],
            TR_result_list:list[dict[str,torch.Tensor]]
        ):
        """
        graph_df (with source) 하나 -> sample_loader 하나
        sample: 
            {
                "src":src,
                "dst":dst,
                "label":label,
                "query_t":query_t,
                "pos_mask":pos_mask,
                "weight":weight
            }

        Input:
            n_node
            graph_loader_list: all type graph_loader list
            TR_result_list: all type TR_result list
        """
        sample_loader_list=[]
        for graph_loader,TR_result in zip(graph_loader_list,TR_result_list):
            source=random.randint(1,n_node)
            sample_loader=TrainUtils.get_TR_sample_loader(
                source=source,
                data_loader=graph_loader,
                TR_result=TR_result,
                sampling=f"focused"
            )
            sample_loader_list.append(sample_loader)
        return sample_loader_list