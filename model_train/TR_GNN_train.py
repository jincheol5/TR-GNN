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
            val_sample_loader:list[dict[str,torch.Tensor]],
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

            ### Epoch마다 train_sample_loader 생성
            train_sample_loader=TrainUtils.get_TR_sample_loader(
                source=kwargs["source"],
                data_loader=train_loader,
                TR_result=TR_result,
                sampling=kwargs["sampling"],
                n_sample=kwargs.get("n_sample",200),
                n_pair=kwargs.get("n_pair",10)
            )

            model.train()
            for batch_idx,(batch_event,batch_sample) in enumerate(
                    tqdm(
                        zip(train_loader,train_sample_loader),
                        total=len(train_sample_loader),
                        desc=f"Training epoch {epoch+1}..."
                    )
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

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                sample_weight=batch_sample.get("weight",torch.ones_like(label))
                sample_weight=sample_weight.to(device)

                ### predict of sample
                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss(weight=sample_weight)
                loss=criterion(pred_logit,label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### memory, last_state detach
                model.memory.memory_detach()
                model.last_state.last_state_detach()

                """
                Neural Execution
                """
                ### predict about batch eventstream
                source=kwargs["source"]
                src=torch.full_like(event_dst,fill_value=source,device=device)
                batch_end_t=event_t.max().expand_as(event_t) # Predict every destination at the batch end time.
                pred_dst_logit=model(
                    src=src,
                    dst=event_dst,
                    event_t=batch_end_t
                ) # [B,1]
                pred_dst_logit=pred_dst_logit.squeeze(-1) # -> [B,]

                ### update last_state for all event_dst
                label_state=TR_result["label"][batch_idx,source,event_dst.cpu()].to(device) # [B,]
                model.last_state.update_last_state(
                    dst=event_dst,
                    pred_logit=pred_dst_logit,
                    label_state=label_state,
                    purpose="train"
                )

            """
            Validate Model
            """
            val_result=TR_GNN_Trainer.validate(
                model=model,
                val_loader=val_loader,
                val_sample_loader=val_sample_loader,
                **kwargs
            )
            val_acc=val_result["acc"]
            print(f"Validate ACC using {kwargs['sampling']} TR Sampling: {val_acc}")

            """
            Check Early Stop
            """
            val_loss=val_result["loss"]
            print(f"{epoch+1} epoch Validate Loss: {val_loss}")
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_loss,
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break
        if kwargs["early_stop"] and early_stop.best_state is not None:
            model.load_state_dict(early_stop.best_state)
        return model

    @staticmethod
    def validate(
            model:nn.Module,
            val_loader:DataLoader,
            val_sample_loader:list,
            **kwargs
        ):
        """
        Validate Loss and ACC 계산
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        compute validate loss and acc
        """
        loss_list=[]
        acc_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(val_loader,val_sample_loader),
                    total=len(val_sample_loader),
                    desc=f"Validate..."
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

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)

                """
                Neural Execution
                """
                ### predict about batch eventstream
                source=kwargs["source"]
                src=torch.full_like(event_dst,fill_value=source,device=device)
                batch_end_t=event_t.max().expand_as(event_t) # Predict every destination at the batch end time.
                pred_dst_logit=model(
                    src=src,
                    dst=event_dst,
                    event_t=batch_end_t
                ) # [B,1]
                pred_dst_logit=pred_dst_logit.squeeze(-1) # -> [B,]

                ### update last_state for all event_dst
                model.last_state.update_last_state(
                    dst=event_dst,
                    pred_logit=pred_dst_logit,
                    purpose="test"
                )
        return {
                    "loss":torch.stack(loss_list).mean().item(),
                    "acc":sum(acc_list)/len(acc_list)
                }

    @staticmethod
    def evaluate(
            model:nn.Module,
            test_loader:DataLoader,
            test_sample_loader:list,
            **kwargs
        ):
        """
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        compute test acc
        """
        pred_logit_list=[]
        label_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_loader),
                    total=len(test_sample_loader),
                    desc=f"Compute Test Acc..."
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

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [n_sample,1]

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                pred_logit_list.append(pred_logit.cpu())
                label_list.append(label.cpu())

                """
                Neural Execution
                """
                ### predict about batch eventstream
                source=kwargs["source"]
                src=torch.full_like(event_dst,fill_value=source,device=device)
                batch_end_t=event_t.max().expand_as(event_t) # Predict every destination at the batch end time.
                pred_dst_logit=model(
                    src=src,
                    dst=event_dst,
                    event_t=batch_end_t
                ) # [B,1]
                pred_dst_logit=pred_dst_logit.squeeze(-1) # -> [B,]

                ### update last_state for all event_dst
                model.last_state.update_last_state(
                    dst=event_dst,
                    pred_logit=pred_dst_logit,
                    purpose="test"
                )
        acc=Metric.compute_accuracy(
            pred_logit=torch.cat(pred_logit_list,dim=0),
            label=torch.cat(label_list,dim=0)
        )
        return {
            "acc":acc
        }

    @staticmethod
    def evaluate_hop_range(
            model:nn.Module,
            test_loader:DataLoader,
            test_sample_loader:list,
            **kwargs
        ):
        """
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        compute test acc
        """
        pred_logit_list=[]
        label_list=[]
        hop_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_loader),
                    total=len(test_sample_loader),
                    desc=f"Compute Test Acc..."
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

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                hop=batch_sample["hop"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [n_sample,1]
                pred_logit=pred_logit.squeeze(-1) # [n_sample,]

                ### Collect all samples for positive-only hop range accuracy
                pred_logit_list.append(pred_logit.cpu())
                label_list.append(label.cpu())
                hop_list.append(hop.cpu())

                """
                Neural Execution
                """
                ### predict about batch eventstream
                source=kwargs["source"]
                src=torch.full_like(event_dst,fill_value=source,device=device)
                batch_end_t=event_t.max().expand_as(event_t) # Predict every destination at the batch end time.
                pred_dst_logit=model(
                    src=src,
                    dst=event_dst,
                    event_t=batch_end_t
                ) # [B,1]
                pred_dst_logit=pred_dst_logit.squeeze(-1) # -> [B,]

                ### update last_state for all event_dst
                model.last_state.update_last_state(
                    dst=event_dst,
                    pred_logit=pred_dst_logit,
                    purpose="test"
                )
        ### Compute accuracy across all batches, weighted by sample count
        if not label_list or not any(label.numel() for label in label_list):
            return {"acc":0.0,"range_1_acc":0.0,"range_2_acc":0.0,"range_3_acc":0.0}
        result=Metric.compute_hop_range_accuracy(
            pred_logit=torch.cat(pred_logit_list,dim=0),
            label=torch.cat(label_list,dim=0),
            hop=torch.cat(hop_list,dim=0)
        )
        return {
            "acc":result["acc"],
            "range_1_acc":result["hop_range_1_acc"],
            "range_2_acc":result["hop_range_2_acc"],
            "range_3_acc":result["hop_range_3_acc"]
        }
