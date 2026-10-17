import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper
from model import TR_GNN

class TR_GNN_Trainer_OOD:
    @staticmethod
    def train(
            model:TR_GNN,
            train_graph_list:list,
            val_graph_list:list,
            train_loader_list:list[DataLoader],
            val_loader_list:list[DataLoader],
            val_sample_loader_list:list[list[dict[str,torch.Tensor]]],
            TR_result_list:list[dict[str,torch.Tensor]],
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
            train_sample_loader_list=TrainUtils.get_sample_loader_list_for_train_OOD(
                n_node=kwargs["n_node"],
                all_graph_loader_list=train_loader_list,
                all_TR_result_list=TR_result_list
            )
            model.train()
            for train_graph,train_loader,train_sample_loader,TR_result in tqdm(
                    zip(train_graph_list,train_loader_list,train_sample_loader_list,TR_result_list),
                    total=len(train_graph_list),
                    desc=f"Training epoch {epoch+1} (OOD)..."
                ):
                model.set_graph(graph=train_graph)
                for batch_idx,(batch_event,batch_sample) in enumerate(zip(train_loader,train_sample_loader)):
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

                    with torch.no_grad():
                        """
                        Neural Execution
                        """
                        ### predict about batch eventstream
                        source=int(batch_sample["src"][0].item())
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
            val_result=TR_GNN_Trainer_OOD.validate(
                model=model,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_sample_loader_list=val_sample_loader_list,
                **kwargs
            )
            val_acc=val_result["acc"]
            print(f"Validate ACC: {val_acc}")

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
            model:TR_GNN,
            val_graph_list:list,
            val_loader_list:list[DataLoader],
            val_sample_loader_list:list[list[dict[str,torch.Tensor]]],
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.eval()

        loss_list=[]
        acc_list=[]
        criterion=nn.BCEWithLogitsLoss(reduction="sum")
        with torch.no_grad():
            for val_graph,val_loader,val_sample_loader in tqdm(
                    zip(val_graph_list,val_loader_list,val_sample_loader_list),
                    total=len(val_loader_list),
                    desc="Validate (OOD)..."
                ):
                model.set_graph(graph=val_graph)

                graph_loss_list=[]
                pred_logit_list=[]
                label_list=[]
                graph_sample_count=0
                for batch_event,batch_sample in zip(val_loader,val_sample_loader):
                    event_src,event_dst,event_t,event_edge=batch_event
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)
                    model.update_model_memory(
                        src=event_src.to(device),
                        dst=event_dst.to(device),
                        event_t=event_t.to(device),
                        edge=event_edge.to(device)
                    )

                    if batch_sample["src"].numel()==0:
                        continue

                    src=batch_sample["src"].to(device)
                    dst=batch_sample["dst"].to(device)
                    query_t=batch_sample["query_t"].to(device)
                    label=batch_sample["label"].to(device)
                    pred_logit=model(
                        src=src,
                        dst=dst,
                        event_t=query_t
                    ).squeeze(-1)

                    graph_loss_list.append(criterion(pred_logit,label))
                    graph_sample_count+=label.numel()
                    pred_logit_list.append(pred_logit.cpu())
                    label_list.append(label.cpu())
                    """
                    Neural Execution
                    """
                    ### predict about batch eventstream
                    source=int(batch_sample["src"][0].item())
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
                if graph_sample_count==0:
                    continue
                # Average over samples within each graph, then weight graphs equally.
                loss_list.append(torch.stack(graph_loss_list).sum()/graph_sample_count)
                acc_list.append(Metric.compute_accuracy(
                    pred_logit=torch.cat(pred_logit_list,dim=0),
                    label=torch.cat(label_list,dim=0)
                ))
        if not loss_list:
            raise ValueError("No nonempty validation samples available.")
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":sum(acc_list)/len(acc_list)
        }

    @staticmethod
    def evaluate(
            model:TR_GNN,
            test_graph_list:list,
            test_loader_list:list[DataLoader],
            test_sample_loader_list:list[list[dict[str,torch.Tensor]]],
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.eval()

        graph_acc_list=[]
        with torch.no_grad():
            for test_graph,test_loader,test_sample_loader in tqdm(
                    zip(test_graph_list,test_loader_list,test_sample_loader_list),
                    total=len(test_loader_list),
                    desc="Evaluate (OOD)..."
                ):
                model.set_graph(graph=test_graph)

                pred_logit_list=[]
                label_list=[]
                for batch_event,batch_sample in zip(test_loader,test_sample_loader):
                    event_src,event_dst,event_t,event_edge=batch_event
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)
                    model.update_model_memory(
                        src=event_src.to(device),
                        dst=event_dst.to(device),
                        event_t=event_t.to(device),
                        edge=event_edge.to(device)
                    )

                    if batch_sample["src"].numel()==0:
                        continue

                    src=batch_sample["src"].to(device)
                    dst=batch_sample["dst"].to(device)
                    query_t=batch_sample["query_t"].to(device)
                    label=batch_sample["label"].to(device)
                    pred_logit=model(
                        src=src,
                        dst=dst,
                        event_t=query_t
                    ).squeeze(-1)
                    pred_logit_list.append(pred_logit.cpu())
                    label_list.append(label.cpu())

                    """
                    Neural Execution
                    """
                    ### predict about batch eventstream
                    source=int(batch_sample["src"][0].item())
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
                if not label_list:
                    continue
                # Compute accuracy over all samples in the current graph.
                graph_acc=Metric.compute_accuracy(
                    pred_logit=torch.cat(pred_logit_list,dim=0),
                    label=torch.cat(label_list,dim=0)
                )
                graph_acc_list.append(graph_acc)

        if not graph_acc_list:
            raise ValueError("No nonempty evaluation samples available.")
        # Give each graph equal weight in the final accuracy.
        return {
            "acc":sum(graph_acc_list)/len(graph_acc_list)
        }

