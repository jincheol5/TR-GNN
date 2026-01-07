import math
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auroc,binary_auprc,binary_f1_score

class Metrics:
    @staticmethod
    def compute_TR_loss(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [B,1]
            label: [B,1]
        Output:
            loss scalar tensor: [] (0차원)
        """

        loss=F.binary_cross_entropy_with_logits(logit,label.float(),reduction='mean')
        return loss

    @staticmethod
    def compute_TR_acc(logit_list:list,label_list:list):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            Accuracy
        """
        acc_list=[]
        for logit,label in zip(logit_list,label_list):
            pred=torch.sigmoid(logit) 
            pred_label=(pred>=0.5).float()   
            correct=(pred_label==label).float()
            step_acc=correct.mean()
            acc_list.append(step_acc)
        acc=torch.stack(acc_list).mean() 
        return acc.cpu().item()

    @staticmethod
    def compute_TR_macroF1(logit_list:list,label_list:list,threshold:float=0.0):
        # 1) 리스트를 하나로 쭉 이어 붙이기 (B가 서로 달라도 문제 없음)
        logit_all=torch.cat(logit_list,dim=0).view(-1) # [all_B,]
        label_all=torch.cat(label_list,dim=0).view(-1) # [all_B,]

        # 2) 예측 레이블(positive=1, negative=0)
        pred_label=(logit_all>=threshold).float()

        # --- Positive class F1 ---
        f1_pos=binary_f1_score(logit_all,label_all,threshold=threshold)

        # --- Negative class F1 ---
        inv_pred=1-pred_label # negative=1로 취급
        inv_label=1-label_all # negative=1로 취급

        # negative class에 대한 F1 (positive 역할을 바꿔서 계산)
        f1_neg=binary_f1_score(inv_pred,inv_label,threshold=0.5)

        # 3) macro-F1 = (F1_pos + F1_neg) / 2
        f1_pos=f1_pos.item()
        f1_neg=f1_neg.item()

        # nan 방지 처리
        if math.isnan(f1_pos):
            f1_pos=0.0
        if math.isnan(f1_neg):
            f1_neg=0.0

        macro_f1=(f1_pos+f1_neg)/2.0
        return macro_f1

    @staticmethod
    def compute_TR_AUROC(logit_list:list,label_list:list):
        # 1) 리스트를 하나로 쭉 이어 붙이기 (B가 서로 달라도 문제 없음)
        logit_all=torch.cat(logit_list,dim=0).view(-1) # [all_B,]
        label_all=torch.cat(label_list,dim=0).view(-1) # [all_B,]
        auroc=binary_auroc(logit_all, label_all).item()
        if math.isnan(auroc):
            auroc=0.5 # label이 모두 1이거나 0이면 0.5 반환
        return auroc 

    @staticmethod
    def compute_TR_PRAUC(logit_list:list,label_list:list):
        # 1) 리스트를 하나로 쭉 이어 붙이기 (B가 서로 달라도 문제 없음)
        logit_all=torch.cat(logit_list,dim=0).view(-1) # [all_B,]
        label_all=torch.cat(label_list,dim=0).view(-1) # [all_B,]
        prauc=binary_auprc(logit_all, label_all).item()
        if math.isnan(prauc):
            prauc=0.0 # label이 모두 1이거나 0이면 0.0 반환
        return prauc

    @staticmethod
    def compute_TR_MCC(logit_list:list,label_list:list,eps=1e-8):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            MCC
        """
        # 1) 리스트를 하나로 쭉 이어 붙이기 (B가 서로 달라도 문제 없음)
        logit_all=torch.cat(logit_list,dim=0) # [all_B,1]
        label_all=torch.cat(label_list,dim=0) # [all_B,1]

        # 2) 시그모이드 후 0/1로 변환
        pred=torch.sigmoid(logit_all)
        pred_label=(pred>=0.5).float()

        # 3) TP,TN,FP,FN 계산
        tp=((pred_label==1)&(label_all==1)).sum().float()
        tn=((pred_label==0)&(label_all==0)).sum().float()
        fp=((pred_label==1)&(label_all==0)).sum().float()
        fn=((pred_label==0)&(label_all==1)).sum().float()

        # 4) MCC numerator & denominator
        numerator=(tp*tn)-(fp*fn)
        denominator=torch.sqrt(
            (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        )

        # 5) 분모가 0이면 MCC = 0 처리 (관례적 방식)
        mcc=numerator/(denominator+eps)
        return mcc.cpu().item()