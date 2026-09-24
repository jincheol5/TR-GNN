import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Literal
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from graph import TemporalGraph
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
    def get_TR_result_using_Temporal_BFS(
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
                TR_info=graph.compute_TR_using_Temporal_BFS(
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

    @staticmethod
    def get_TR_sample_list(
            data_loader:DataLoader,
            n_pair:int,
            source:int|None=None,
            TR_result:dict[str,torch.Tensor]|None=None,
            sampling:Literal[
                "independent",
                "dependent",
                "hop_range"
            ]=f"dependent"
        )->list[dict[str,torch.Tensor]]:
        """
        Input:
            data_loader
            n_pair
            source
            TR_result
            sampling
        Return:
            TR_sample_list
        """
        TR_label=TR_result["label"]
        TR_hop=TR_result["hop"]
        TR_sample_list=[]
        for batch_idx,(src,dst,event_t,_) in tqdm(
                enumerate(data_loader),
                desc="Generating TR samples..."
            ):
            sources=torch.unique(torch.cat([src,dst])).tolist()
            query_time=event_t.max().item()
            match sampling:
                case "independent":
                    TR_sample=SamplingUtils.source_independent_TR_sampling(
                        sources=sources,
                        n_pair=n_pair,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx]
                    )
                case "dependent":
                    TR_sample=SamplingUtils.source_dependent_TR_sampling(
                        source=source,
                        n_pair=n_pair,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx],
                        updated_nodes=sources
                    )
                case "hop_range":
                    TR_sample=SamplingUtils.hop_range_TR_sampling(
                        source=source,
                        n_pair=n_pair,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx],
                        TR_hop=TR_hop[batch_idx]
                    )
            TR_sample_list.append(TR_sample)
        return TR_sample_list

    @staticmethod
    def get_source_candidates(
            n_source:int,
            TR_label:torch.Tensor
        )->list[int]:
        """
        seq별 source의 reachability ratio를 계산하여
        다음 조건을 모두 만족하는 source 후보를 반환.

        조건:
            - seq 평균 reachability ratio: 40% <= mean <= 60%
            - padding node(id=0)는 source/destination 후보에서 제외.

        Input:
            TR_label: [seq_len,N+1,N+1] bool tensor
        Return:
            source_candidates: list[int]
        """
        _,n_node,_=TR_label.shape
        n_node=n_node-1  # padding node 제외한 실제 node 수

        ### padding source/destination 제거
        # [seq_len,N,N]
        label=TR_label[:,1:,1:]

        ### reachable node 개수
        # 자기 자신 포함
        # [seq_len,N]
        reachable_count=label.sum(dim=2)

        ### seq별 source reachability ratio
        # [seq_len,N]
        reachability_ratio=(reachable_count.float()/n_node)

        ### source별 seq 통계
        # [N]
        mean_ratio=reachability_ratio.mean(dim=0)

        ### 조건 적용
        candidate_mask=(
            (mean_ratio>=0.4) &
            (mean_ratio<=0.6) 
        )

        ### 실제 source node id
        # index 0 -> node id 1
        source_candidates=torch.where(candidate_mask)[0]+1

        ### 후보 중 n_source개 random sampling
        if source_candidates.numel()>n_source:
            perm=torch.randperm(source_candidates.numel())
            source_candidates=source_candidates[perm[:n_source]]

        ### list[int] 변환
        source_candidates=(
            source_candidates
            .cpu()
            .tolist()
        )
        return source_candidates