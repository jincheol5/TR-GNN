import os
import random
import threading
import queue
import wandb
import torch
import numpy as np
from typing_extensions import Literal
from tqdm import tqdm
from .data_utils import DataUtils
from .model_train_utils import ModelTrainUtils,EarlyStopping
from .metrics import Metrics

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader_list,val_data_loader_list,validate:bool=False,config:dict=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        """
        Early stopping
        """
        if config['early_stop']:
            early_stop=EarlyStopping(patience=config['patience'])

        """
        model train
        """
        for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
            loss_list=[]
            model.train()
            for data_loader in tqdm(train_data_loader_list,desc=f"Epoch {epoch+1}..."):
                label_list=[batch['label'] for batch in data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
                label_list=[label.to(device) for label in label_list]

                output=model(data_loader=data_loader,device=device,mode='train') # List of [B,1], B는 각 element마다 다를 수 있음

                loss=Metrics.compute_tR_loss(logit_list=output,label_list=label_list)
                loss_list.append(loss)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss=torch.stack(loss_list).mean().item()

            """
            Early stopping
            """
            if config['early_stop']:
                val_loss=epoch_loss
                pre_model=early_stop(val_loss=val_loss,model=model)
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stopping in epoch {epoch+1}")
                    break

            """
            wandb log
            """
            if config['wandb']:
                wandb.log({
                    f"loss":epoch_loss,
                },step=epoch)
            
            """
            validate
            """
            if validate:
                acc,macrof1,auroc,prauc,mcc=ModelTrainer.test(model=model,data_loader_list=val_data_loader_list)
                print(f"{epoch+1} epoch tR validation Acc: {acc} macro-f1: {macrof1} AUROC: {auroc} PR-AUC: {prauc} MCC: {mcc}")

    @staticmethod
    def test(model,data_loader_list):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        """
        model test
        """
        acc_list=[]
        all_logit_list=[]
        all_label_list=[]
        with torch.no_grad():
            for data_loader in tqdm(data_loader_list,desc=f"Evaluating..."):
                label_list=[batch['label'] for batch in data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
                label_list=[label.to(device) for label in label_list]
                
                output=model(data_loader=data_loader,device=device,mode='test') # List of [B,1], B는 각 element마다 다를 수 있음

                acc=Metrics.compute_tR_acc(logit_list=output,label_list=label_list)
                acc_list.append(acc)

                all_logit_list+=output
                all_label_list+=label_list

        # compute macrof1,auroc,prauc,mcc
        acc=float(np.mean(acc_list))
        macrof1=Metrics.compute_tR_macroF1(logit_list=all_logit_list,label_list=all_label_list)
        auroc=Metrics.compute_tR_AUROC(logit_list=all_logit_list,label_list=all_label_list)
        prauc=Metrics.compute_tR_PRAUC(logit_list=all_logit_list,label_list=all_label_list)
        mcc=Metrics.compute_tR_MCC(logit_list=all_logit_list,label_list=all_label_list)
        return acc,macrof1,auroc,prauc,mcc

    @staticmethod
    def test_chunk(model,config:dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        chunk_dir_path=os.path.join('..','data','trgnn','test',f"{config['num_nodes']}")
        chunk_files=sorted(
            [f for f in os.listdir(chunk_dir_path) if f.startswith(f"test_{config['num_nodes']}_chunk_{config['chunk_size']}_")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])  # 마지막 index 숫자로 정렬
        )
        chunk_paths=[os.path.join(chunk_dir_path,f) for f in chunk_files]
        num_chunks=len(chunk_paths) # for tqdm

        buffer_queue=queue.Queue(maxsize=2)
        loader_thread=threading.Thread(
            target=ModelTrainUtils.chunk_loader_worker,
            args=(chunk_paths,buffer_queue)
        )
        loader_thread.start()

        acc_list=[]
        all_logit_list=[]
        all_label_list=[]
        with torch.no_grad():
            pbar=tqdm(total=num_chunks,desc="Evaluating chunks...") # tqdm: 전체 chunk 수 기준
            while True:
                dataset_list=buffer_queue.get()
                if dataset_list is None:
                    break

                data_loader_list=[]
                for dataset in dataset_list:
                    data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                    data_loader_list.append(data_loader)
                
                for data_loader in data_loader_list:
                    label_list=[batch['label'] for batch in data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
                    label_list=[label.to(device) for label in label_list]

                    output=model(data_loader=data_loader,device=device,mode='test') # List of [B,1], B는 각 element마다 다를 수 있음

                    acc=Metrics.compute_tR_acc(logit_list=output,label_list=label_list)
                    acc_list.append(acc)

                    all_logit_list+=output
                    all_label_list+=label_list
                pbar.update(1) # chunk 처리 완료 → tqdm 1 증가

                # 메모리 정리
                del dataset_list
                del data_loader_list
                del data_loader

        # compute macrof1,auroc,prauc,mcc
        acc=float(np.mean(acc_list))
        macrof1=Metrics.compute_tR_macroF1(logit_list=all_logit_list,label_list=all_label_list)
        auroc=Metrics.compute_tR_AUROC(logit_list=all_logit_list,label_list=all_label_list)
        prauc=Metrics.compute_tR_PRAUC(logit_list=all_logit_list,label_list=all_label_list)
        mcc=Metrics.compute_tR_MCC(logit_list=all_logit_list,label_list=all_label_list)
        return acc,macrof1,auroc,prauc,mcc