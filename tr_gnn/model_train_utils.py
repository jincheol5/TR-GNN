import pickle
import queue
import numpy as np
import torch

class ModelTrainUtils:
    @staticmethod
    def get_data_loader(datastream:dict,traj:torch.Tensor,source_id:int=0,batch_size:int=1):
        """
        Input:
            datastream: dict
                mem_t: [E,N,1]
                emb_t: [E,N,1]
                src: [E,1]
                tar: [E,1]
                n_mask: [E,N]
            traj: [E,N,1]
            source_id: source node id
            batch_size: batch size of edge_events
        Output:
            data_loader: batch list
                batch: dict
                    traj: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N]
                    label: [B,1]
        """
        E,num_nodes=datastream['n_mask'].size()
        init_traj=torch.zeros((num_nodes,1),dtype=torch.float)
        init_traj[source_id,0]=1.0

        data_loader=[]
        prev_mem_block=None  # store mem_t from previous batch to delay mem by one batch
        for start in range(0,E,batch_size):
            end=min(start+batch_size,E)
            batch={}
            batch['init_traj']=init_traj # [N,1]
            batch['traj']=traj[start:end] # [B,N,1]
            batch['emb_t']=datastream['emb_t'][start:end] # [B,N,1]
            current_mem=datastream['mem_t'][start:end] # [B,N,1]
            if prev_mem_block is None:
                batch['mem_t']=torch.zeros_like(current_mem) # [B,N,1] (first batch all zeros)
            else:
                batch['mem_t']=prev_mem_block # [B,N,1] delayed by one batch
            prev_mem_block=current_mem
            batch['src']=datastream['src'][start:end] # [B,1]
            batch['tar']=datastream['tar'][start:end] # [B,1] 
            batch['n_mask']=datastream['n_mask'][start:end] # [B,N]

            # label: trajectory value for the target node of each event
            # batch['traj']: [B,N,1], batch['tar']: [B,1]
            tar_idx=batch['tar'].squeeze(-1).long()  # [B]
            batch_traj=batch['traj'] # [B,N,1]
            # select per-row the value at the target node -> result [B,1]
            labels=batch_traj[torch.arange(batch_traj.size(0)),tar_idx] # [B,1]
            batch['label']=labels # [B,1]

            data_loader.append(batch)
        return data_loader

    @staticmethod
    def teacher_forcing(pred:torch.Tensor,label:torch.Tensor,p:float=0.5):
        """
        Input
            pred: [B,1]
            label: [B,1]
            p: float
        Output:
            updated_r_pred
        """
        batch_size=pred.size(0)
        mask=(torch.rand(batch_size,device=pred.device)<p).unsqueeze(-1) # [B,1] boolean mask
        pred=torch.where(mask,label,pred) # teacher forcing
        return pred

    @staticmethod
    def chunk_loader_worker(chunk_paths:str,buffer_queue:queue.Queue):
        """
        chunk_paths: chunk 파일 리스트
        """
        print(f"Run chunk_loader_worker!")
        for path in chunk_paths:
            with open(path,"rb") as f:
                data=pickle.load(f)
            buffer_queue.put(data) # 버퍼가 꽉 차면 자동 대기
        buffer_queue.put(None) # 종료 신호

class EarlyStopping:
    def __init__(self,patience=1):
        self.patience=patience
        self.patience_count=0
        self.prev_acc=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,val_acc:float,model:torch.nn.Module):
        if self.prev_acc==np.inf:
            self.prev_acc=val_acc
            self.best_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        else:
            if not np.isfinite(val_acc):
                print(f"Acc is NaN or Inf!")
                self.early_stop=True
                model.load_state_dict(self.best_state)
                return model
            if self.prev_acc>=val_acc:
                self.patience_count+=1
                if self.patience<self.patience_count:
                    print(f"Acc decreases during {self.patience_count} patience!")
                    self.early_stop=True
                    model.load_state_dict(self.best_state)
                    return model
            else:
                self.patience_count=0
                self.prev_acc=val_acc
                self.best_state={k: v.clone() for k,v in model.state_dict().items()}
                return None