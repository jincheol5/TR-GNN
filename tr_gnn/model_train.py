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
    def train(model,train_data_loader_list:list=None,val_data_loader_list:list=None,config:dict=None):
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
            for train_data_loader in tqdm(train_data_loader_list,desc=f"Training epoch: {epoch}..."):
                label_list=[batch['label'] for batch in train_data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
                label_list=[label.to(device) for label in label_list]

                logit_list=model(data_loader=train_data_loader,device=device,mode="train")
                loss=Metrics.compute_TR_loss(logit_list=logit_list,label_list=label_list)
                loss_list.append(loss)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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
            perform=ModelTrainer.test(model=model,data_loader_list=val_data_loader_list)
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
    def test(model,data_loader_list=None):
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
                label_list=[batch['label'] for batch in data_loader]
                label_list=[label.to(device) for label in label_list]
                
                logit_list=model(data_loader=data_loader,device=device,mode="test")

                acc=Metrics.compute_TR_acc(logit_list=logit_list,label_list=label_list)
                acc_list.append(acc)

                all_logit_list+=logit_list
                all_label_list+=label_list

        # compute acc,macrof1,auroc,prauc,mcc
        perform={
            'acc':float(np.mean(acc_list)),
            'macrof1':Metrics.compute_TR_macroF1(logit_list=all_logit_list,label_list=all_label_list),
            'prauc':Metrics.compute_TR_PRAUC(logit_list=all_logit_list,label_list=all_label_list),
            'mcc':Metrics.compute_TR_MCC(logit_list=all_logit_list,label_list=all_label_list)
        }
        return perform
