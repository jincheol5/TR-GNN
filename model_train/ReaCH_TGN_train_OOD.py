import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper,ReaCH_TGN_Utils
from model import ReaCH_TGN

class ReaCH_TGN_Trainer_OOD:
    @staticmethod
    def train(
            model:ReaCH_TGN,
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
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
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
                train_graph.set_source_info(
                    source=int(train_sample_loader[0]["src"][0].item())
                )
                TR_hop=TR_result["hop"]
                TR_first_t=TR_result["first_t"]
                for batch_idx,(batch_event,batch_sample) in enumerate(zip(train_loader,train_sample_loader)):
                    ### Eventstream
                    event_src,event_dst,event_t,event_edge=batch_event
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)

                    ### Generate view A,B
                    view_A_event_t=ReaCH_TGN_Utils.augment_event_t(event_t=event_t)
                    view_A_updated_mem_vec=model.get_updated_mem_vec(
                        src=event_src,
                        dst=event_dst,
                        edge=event_edge,
                        event_t=view_A_event_t
                    )
                    view_A_embedding_result=model.get_embedding_result(
                        src=event_src,
                        dst=event_dst,
                        event_t=view_A_event_t,
                        mem_vec=view_A_updated_mem_vec
                    )
                    view_A_src_vec=view_A_embedding_result["src_vec"]
                    view_A_dst_vec=view_A_embedding_result["dst_vec"]

                    view_B_event_t=ReaCH_TGN_Utils.augment_event_t(event_t=event_t)
                    view_B_updated_mem_vec=model.get_updated_mem_vec(
                        src=event_src,
                        dst=event_dst,
                        edge=event_edge,
                        event_t=view_B_event_t
                    )
                    view_B_embedding_result=model.get_embedding_result(
                        src=event_src,
                        dst=event_dst,
                        event_t=view_B_event_t,
                        mem_vec=view_B_updated_mem_vec
                    )
                    view_B_src_vec=view_B_embedding_result["src_vec"]
                    view_B_dst_vec=view_B_embedding_result["dst_vec"]

                    ### Compute NT_Xent Loss
                    NT_Xent_loss=ReaCH_TGN_Utils.compute_NT_Xent_Loss(
                        Z_src_A=view_A_src_vec,
                        Z_dst_A=view_A_dst_vec,
                        Z_src_B=view_B_src_vec,
                        Z_dst_B=view_B_dst_vec
                    )

                    ### update memory for eventsream
                    model.update_model_memory(
                        src=event_src,
                        dst=event_dst,
                        event_t=event_t,
                        edge=event_edge
                    )

                    ### batch TR result
                    batch_TR_hop=TR_hop[batch_idx]
                    batch_TR_first_t=TR_first_t[batch_idx]
                    batch_TR_hop=batch_TR_hop.to(device)
                    batch_TR_first_t=batch_TR_first_t.to(device)

                    ### TR Sample
                    src=batch_sample["src"]
                    dst=batch_sample["dst"]
                    query_t=batch_sample["query_t"]
                    label=batch_sample["label"]
                    pos_mask=batch_sample["pos_mask"]
                    src=src.to(device)
                    dst=dst.to(device)
                    query_t=query_t.to(device)
                    label=label.to(device)
                    pos_mask=pos_mask.to(device)
                    sample_weight=batch_sample["weight"]
                    sample_weight=sample_weight.to(device)

                    pred_logit=model(
                        src=src,
                        dst=dst,
                        event_t=query_t
                    ) # [B,1]

                    ### Hop-Based Penalty
                    # pos_pair에 대한 penalty 계산
                    pos_src=src[pos_mask]
                    pos_dst=dst[pos_mask]
                    pos_pair_hop=batch_TR_hop[pos_src,pos_dst]
                    weight_hop=ReaCH_TGN_Utils.compute_hop_based_penalty(
                        pair_hop=pos_pair_hop,
                        max_hop=batch_TR_hop.max().item()
                    )

                    ### Time-Gap Penalty
                    # pos_pair에 대한 penalty 계산
                    query_time=event_t.max().item()
                    pos_pair_first_t=batch_TR_first_t[pos_src,pos_dst]
                    weight_time=ReaCH_TGN_Utils.compute_time_gap_penalty(
                        pair_first_t=pos_pair_first_t,
                        query_time=query_time
                    )

                    ### Pair Weight
                    pair_weight=torch.ones_like(label)
                    pair_weight[pos_mask]=weight_hop*weight_time

                    ### Weighted BCE Loss
                    pred_logit=pred_logit.squeeze(-1) # -> [B,]
                    final_weight=sample_weight*pair_weight
                    criterion=nn.BCEWithLogitsLoss(weight=final_weight)
                    loss=criterion(pred_logit,label)
                    total_loss=loss+NT_Xent_loss

                    ### backward
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    ### memory detach
                    model.memory.memory_detach()
            """
            Validate Model
            """
            val_result=ReaCH_TGN_Trainer_OOD.validate(
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
            model:ReaCH_TGN,
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
                source=next(
                    (int(sample["src"][0].item()) for sample in val_sample_loader
                     if sample["src"].numel()>0),
                    None
                )
                if source is None:
                    continue
                val_graph.set_source_info(source=source)

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
            model:ReaCH_TGN,
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
                source=next(
                    (int(sample["src"][0].item()) for sample in test_sample_loader
                     if sample["src"].numel()>0),
                    None
                )
                if source is None:
                    continue
                test_graph.set_source_info(source=source)

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

