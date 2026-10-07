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
            train_sample_list=TrainUtils.get_TR_sample_list(
                data_loader=train_loader,
                n_sample=kwargs["n_sample"],
                TR_result=TR_result,
                source=kwargs["source"],
                sampling="focused"
            )

            model.train()
            for batch_idx,(batch_event,batch_sample) in enumerate(
                    tqdm(
                        zip(train_loader,train_sample_list),
                        total=len(train_sample_list),
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

                sample_weight=batch_sample["weight"]
                sample_weight=sample_weight.to(device)

                ### predict of sample
                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Loss
                # pred_logit=pred_logit.squeeze(-1) # -> [B,]
                # criterion=nn.BCEWithLogitsLoss()
                # loss=criterion(pred_logit,label)
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                if kwargs["sampling"]=="focused":
                    criterion=nn.BCEWithLogitsLoss(weight=sample_weight)
                else: # random
                    criterion=nn.BCEWithLogitsLoss()
                # criterion=nn.BCEWithLogitsLoss()
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
                val_sample_list=val_sample_list,
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
        return model

    @staticmethod
    def validate(
            model:nn.Module,
            val_loader:DataLoader,
            val_sample_list:DataLoader,
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
                    zip(val_loader,val_sample_list),
                    total=len(val_sample_list),
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
            test_sample_list:list,
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
        acc_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_list),
                    total=len(test_sample_list),
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
            "acc":sum(acc_list)/len(acc_list)
        }

    @staticmethod
    def evaluate_hop_range(
            model:nn.Module,
            test_loader:DataLoader,
            test_sample_list:list,
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
        range_1_acc_list=[]
        range_2_acc_list=[]
        range_3_acc_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_list),
                    total=len(test_sample_list),
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
                range_1_mask=batch_sample["range_1_mask"]
                range_2_mask=batch_sample["range_2_mask"]
                range_3_mask=batch_sample["range_3_mask"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)
                range_1_mask=range_1_mask.to(device)
                range_2_mask=range_2_mask.to(device)
                range_3_mask=range_3_mask.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [n_sample,1]
                pred_logit=pred_logit.squeeze(-1) # [n_sample,]

                ### Range 1 Accuracy
                if range_1_mask.any():
                    range_1_acc=Metric.compute_accuracy(
                        pred_logit=pred_logit[range_1_mask],
                        label=label[range_1_mask]
                    )
                    range_1_acc_list.append(range_1_acc)

                ### Range 2 Accuracy
                if range_2_mask.any():
                    range_2_acc=Metric.compute_accuracy(
                        pred_logit=pred_logit[range_2_mask],
                        label=label[range_2_mask]
                    )
                    range_2_acc_list.append(range_2_acc)

                ### Range 3 Accuracy
                if range_3_mask.any():
                    range_3_acc=Metric.compute_accuracy(
                        pred_logit=pred_logit[range_3_mask],
                        label=label[range_3_mask]
                    )
                    range_3_acc_list.append(range_3_acc)

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
        ### Average Accuracy
        range_1_acc=sum(range_1_acc_list)/len(range_1_acc_list) if len(range_1_acc_list)>0 else 0.0
        range_2_acc=sum(range_2_acc_list)/len(range_2_acc_list) if len(range_2_acc_list)>0 else 0.0
        range_3_acc=sum(range_3_acc_list)/len(range_3_acc_list) if len(range_3_acc_list)>0 else 0.0
        return {
            "range_1_acc":range_1_acc,
            "range_2_acc":range_2_acc,
            "range_3_acc":range_3_acc
        }