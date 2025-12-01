import pickle
import queue
import numpy as np
import torch

class ModelTrainUtils:
    @staticmethod
    def get_data_loader(dataset:dict,batch_size:int):
        """
        Input:
            dataset:
                raw: [seq_len,N,1]
                r: [seq_len,N,1]
                emb_t: [seq_len,N,1]
                mem_t: [seq_len,N,1]
                src: [seq_len,1]
                tar: [seq_len,1]
                n_mask: [seq_len,N]
                label: [seq_len,1]
            batch_size: batch size of edge_events
        Output:
            data_loader: 
                batch
                    raw: [B,N,1]
                    r: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N]
                    label: [B,1]
        """
        seq_len=dataset['r'].size(0)
        data_loader=[]
        for start in range(0,seq_len,batch_size):
            end=start+batch_size
            batch={
                "raw":dataset["raw"][start:end], # [B,N,1]
                "r":dataset["r"][start:end], # [B,N,1]
                "emb_t":dataset["emb_t"][start:end], # [B,N,1]
                "mem_t":dataset["mem_t"][start:end], # [B,N,1]
                "src":dataset["src"][start:end], # [B,1]
                "tar":dataset["tar"][start:end], # [B,1]
                "n_mask":dataset["n_mask"][start:end], # [B,N]
                "label":dataset["label"][start:end], # [B,1]
            }
            data_loader.append(batch)
        return data_loader

    @staticmethod
    def teacher_forcing(r_pred:torch.Tensor,r_label:torch.Tensor,tar:torch.Tensor,p:float=0.5):
        """
        Input
            r_pred: [N,1]
            r_label: [N,1]
            tar: [B,]
            p: float
        Output:
            updated_r_pred
        """
        batch_size=tar.size(0)
        mask=(torch.rand(batch_size,device=r_pred.device)<p) # [B,], boolean tensor
        for i in range(batch_size):
            tar_id=tar[i]
            if mask[i]:
                r_pred[tar_id]=r_label[tar_id]
        return r_pred

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
        self.prev_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,val_loss:float,model:torch.nn.Module):
        if self.prev_loss==np.inf:
            self.prev_loss=val_loss
            self.best_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        else:
            if not np.isfinite(val_loss):
                print(f"Loss is NaN or Inf!")
                self.early_stop=True
                model.load_state_dict(self.best_state)
                return model
            
            if self.prev_loss<=val_loss:
                self.patience_count+=1
                if self.patience<self.patience_count:
                    print(f"Loss increases during {self.patience_count} patience!")
                    self.early_stop=True
                    model.load_state_dict(self.best_state)
                    return model
            else:
                self.patience_count=0
                self.prev_loss=val_loss
                self.best_state={k: v.clone() for k,v in model.state_dict().items()}
                return None