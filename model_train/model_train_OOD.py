import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper

class ModelTrainer_OOD:
    @staticmethod
    def train(
            model:nn.Module,
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
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training (OOD)..."):
            ### Epoch마다 train_sample_loader_list 생성
            train_sample_loader_list=TrainUtils.get_sample_loader_list(
                n_node=kwargs["n_node"],
                graph_loader_list=train_loader_list,
                TR_result_list=TR_result_list
            )

            model.train()
            for train_graph,train_loader,train_sample_loader in tqdm(
                    zip(train_graph_list,train_loader_list,train_sample_loader_list),
                    total=len(train_loader_list),
                    desc=f"Training epoch {epoch+1}..."
                ):
                model.set_graph(graph=train_graph)

                for batch_event,batch_sample in tqdm(
                        zip(train_loader,train_sample_loader),
                        total=len(train_sample_loader),
                        desc=f"Train each graph..."
                    ):
                    ### Update model memory for Eventstream
                    event_src,event_dst,event_t,event_edge=batch_event
                    if kwargs["model_name"] in ("TGN"):
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
                    sample_weight=None
                    if kwargs["sampling"]=="focused":
                        sample_weight=batch_sample["weight"]
                        sample_weight=sample_weight.to(device)

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

                    ### memory detach
                    if kwargs["model_name"] in ("TGN"):
                        model.memory.memory_detach()

            """
            Validate Model
            """
            val_result=ModelTrainer_OOD.validate(
                model=model,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_sample_loader_list=val_sample_loader_list,
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
        model.graph.to_device(device=device)
        model.eval()

        loss_list=[]
        acc_list=[]
        criterion=nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for val_graph,val_loader,val_sample_loader in tqdm(
                    zip(val_graph_list,val_loader_list,val_sample_loader_list),
                    total=len(val_loader_list),
                    desc="Validate (OOD)..."
                ):
                model.set_graph(graph=val_graph)

                graph_loss_list=[]
                graph_acc_list=[]
                for batch_event,batch_sample in tqdm(
                        zip(val_loader,val_sample_loader),
                        total=len(val_sample_loader),
                        desc="Validate each graph..."
                    ):
                    event_src,event_dst,event_t,event_edge=batch_event
                    if kwargs["model_name"] == "TGN":
                        model.update_model_memory(
                            src=event_src.to(device),
                            dst=event_dst.to(device),
                            event_t=event_t.to(device),
                            edge=event_edge.to(device)
                        )

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
                    graph_acc_list.append(Metric.compute_accuracy(
                        pred_logit=pred_logit,
                        label=label
                    ))
                loss_list.append(torch.stack(graph_loss_list).mean())
                acc_list.append(sum(graph_acc_list)/len(graph_acc_list))
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":sum(acc_list)/len(acc_list)
        }

    @staticmethod
    def evaluate(
            model:nn.Module,
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
        model.graph.to_device(device=device)
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
                for batch_event,batch_sample in tqdm(
                        zip(test_loader,test_sample_loader),
                        total=len(test_sample_loader),
                        desc="Evaluate each graph..."
                    ):
                    event_src,event_dst,event_t,event_edge=batch_event
                    if kwargs["model_name"]=="TGN":
                        model.update_model_memory(
                            src=event_src.to(device),
                            dst=event_dst.to(device),
                            event_t=event_t.to(device),
                            edge=event_edge.to(device)
                        )

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
                # Compute accuracy over all samples in the current graph.
                graph_acc=Metric.compute_accuracy(
                    pred_logit=torch.cat(pred_logit_list,dim=0),
                    label=torch.cat(label_list,dim=0)
                )
                graph_acc_list.append(graph_acc)

        # Give each graph equal weight in the final accuracy.
        return {
            "acc":sum(graph_acc_list)/len(graph_acc_list)
        }

