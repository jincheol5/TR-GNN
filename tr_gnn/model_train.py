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
    def train(model,is_memory:bool=False,train_data_loader_list:list=None,val_data_loader_list:list=None,config:dict=None):
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
            model.train()
            epoch_loss_list=[]
            for train_data_loader in tqdm(train_data_loader_list,desc=f"Training epoch: {epoch}..."):
                loss_list=[]
                memory=None
                for batch in train_data_loader:
                    # move batch tensors to device so model and loss use same device
                    batch={k:v.to(device) for k,v in batch.items()}
                    if is_memory:
                        logit,memory=model(batch=batch,pre_memory=memory,device=device)
                    else:
                        logit=model(batch=batch,device=device)
                    loss=Metrics.compute_TR_loss(logit=logit,label=batch['label'])
                    loss_list.append(loss)

                    # back propagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if is_memory:
                        memory=memory.detach()
                epoch_loss_list.append(torch.stack(loss_list).mean().item())
            """
            wandb log
            """
            epoch_loss=np.mean(epoch_loss_list)
            if config['wandb']:
                wandb.log({
                    f"loss":epoch_loss,
                },step=epoch)
            
            """
            validate
            """
            perform=ModelTrainer.test(model=model,is_memory=is_memory,data_loader_list=val_data_loader_list)
            print(f"{epoch+1} epoch TR validation Acc: {perform['acc']} Macro-f1: {perform['macrof1']} PR-AUC: {perform['prauc']} MCC: {perform['mcc']}")
            
            """
            Early stopping
            """
            if config['early_stop']:
                val_acc=perform['acc']
                pre_model=early_stop(val_acc=val_acc,model=model)
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stopping in epoch {epoch+1}")
                    break

    @staticmethod
    def test(model,is_memory:bool=False,data_loader_list=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        """
        model test
        """
        acc_list=[]
        macrof1_list=[]
        prauc_list=[]
        mcc_list=[]
        with torch.no_grad():
            for data_loader in tqdm(data_loader_list,desc=f"Evaluating..."):
                logit_list=[]
                label_list=[]
                memory=None
                for batch in data_loader:
                    # move batch tensors to device so model and metrics use same device
                    batch={k:v.to(device) for k,v in batch.items()}
                    if is_memory:
                        logit,memory=model(batch=batch,pre_memory=memory,device=device)
                    else:
                        logit=model(batch=batch,device=device)
                    logit_list.append(logit)
                    label_list.append(batch['label'])
                acc_list.append(Metrics.compute_TR_acc(logit_list=logit_list,label_list=label_list))
                macrof1_list.append(Metrics.compute_TR_macroF1(logit_list=logit_list,label_list=label_list))
                prauc_list.append(Metrics.compute_TR_PRAUC(logit_list=logit_list,label_list=label_list))
                mcc_list.append(Metrics.compute_TR_MCC(logit_list=logit_list,label_list=label_list))
        perform={
            'acc':np.mean(acc_list),
            'macrof1':np.mean(macrof1_list),
            'prauc':np.mean(prauc_list),
            'mcc':np.mean(mcc_list)
        }
        return perform