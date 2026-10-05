import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper
from model import TR_GNN

class TR_GNN_Trainer:
    @staticmethod
    def train(
            model:TR_GNN,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_list:list[dict[str,torch.Tensor]],
            TR_result:dict[str,torch.Tensor],
            **kwargs
        ):
        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        Model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"TR-GNN Training..."):
            ### Epoch마다 memory state, last_state 초기화
            model.memory.init_memory_state()
            model.last_state.init_last_state()

            ### Epoch마다 train_sample_list 생성
            train_sample_list=[]

            model.train()
            for batch_event,batch_sample in tqdm(
                    zip(train_loader,train_sample_list),
                    total=len(train_sample_list),
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### Update model memory for Eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_model_memory(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### predict
                pred_logit=model(
                    src=event_src,
                    dst=event_dst,
                    event_t=query_t
                ) # [B,1]

                ### update last_state for all event_dst
                

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                ### predict of sample
                pred_sample_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Loss
                pred_sample_logit=pred_sample_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_sample_logit,label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### memory, last_state detach
                model.memory.memory_detach()
                model.last_state.last_state_detach()