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
    def train(model,is_memory:bool=False,train_data_loader:list=None,val_data_loader_list:list=None,validate:bool=False,config:dict=None):
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
            loss_list=[]
            memory=None
            for batch in tqdm(train_data_loader,desc=f"Epoch {epoch+1}..."):
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

            """
            wandb log
            """
            epoch_loss=torch.stack(loss_list).mean().item()
            if config['wandb']:
                wandb.log({
                    f"loss":epoch_loss,
                },step=epoch)
            
            """
            validate
            """
            if validate:
                val_acc_list=[]
                val_macrof1_list=[]
                val_prauc_list=[]
                val_mcc_list=[]
                for val_data_loader in val_data_loader_list:
                    perform=ModelTrainer.test(model=model,is_memory=is_memory,data_loader=val_data_loader)
                    val_acc_list.append(perform['acc'])
                    val_macrof1_list.append(perform['macrof1'])
                    val_prauc_list.append(perform['prauc'])
                    val_mcc_list.append(perform['mcc'])
                print(f"{epoch+1} epoch TR validation Acc: {np.mean(val_acc_list)} macro-f1: {np.mean(val_macrof1_list)} PR-AUC: {np.mean(val_prauc_list)} MCC: {np.mean(val_mcc_list)}")
            
            """
            Early stopping
            """
            if config['early_stop']:
                val_acc=np.mean(val_acc_list)
                pre_model=early_stop(val_acc=val_acc,model=model)
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stopping in epoch {epoch+1}")
                    break

    @staticmethod
    def test(model,is_memory:bool=False,data_loader=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        """
        model test
        """
        logit_list=[]
        label_list=[]
        memory=None
        with torch.no_grad():
            for batch in tqdm(data_loader,desc=f"Evaluating..."):
                # move batch tensors to device so model and metrics use same device
                batch={k:v.to(device) for k,v in batch.items()}
                if is_memory:
                    logit,memory=model(batch=batch,pre_memory=memory,device=device)
                else:
                    logit=model(batch=batch,device=device)
                logit_list.append(logit)
                label_list.append(batch['label'])
        
        perform={
            'acc':Metrics.compute_TR_acc(logit_list=logit_list,label_list=label_list),
            'macrof1':Metrics.compute_TR_macroF1(logit_list=logit_list,label_list=label_list),
            'prauc':Metrics.compute_TR_PRAUC(logit_list=logit_list,label_list=label_list),
            'mcc':Metrics.compute_TR_MCC(logit_list=logit_list,label_list=label_list)
        }
        return perform