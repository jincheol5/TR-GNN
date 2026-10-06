import torch

class SamplingUtils:
    @staticmethod
    def source_random_TR_sampling(
            n_sample:int,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        source를 랜덤 순회하며 총 n_sample개의 positive pair + negative pair (1:1비율)를 생성한다.
        source마다 양성/음성을 각각 최대 n_pair개, 작은 쪽에 맞춰 동일하게 선택한다.
        한쪽 후보가 없으면 건너뛰고 padding, 자기 자신, 중복 dst는 제외한다.
        후보가 소진되면 가능한 개수만 반환한다. 홀수 n_sample은 짝수로 내림한다.

        Input:
            n_sample
            n_pair
            query_time
            TR_label: [N+1,N+1] bool tensor
        Return:
            tensor dict:
                src
                dst
                label
                query_t
                pos_mask
        """
        pairs=[]
        remaining=n_sample
        for source in (torch.randperm(TR_label.size(0)-1)+1).tolist():
            if remaining<2:
                break
            pos=torch.where(TR_label[source])[0]
            neg=torch.where(~TR_label[source])[0]
            pos=pos[(pos!=0) & (pos!=source)]
            neg=neg[(neg!=0) & (neg!=source)]
            count=min(n_pair,len(pos),len(neg),remaining//2)
            if count<=0:
                continue

            pos=pos[torch.randperm(len(pos))[:count]]
            neg=neg[torch.randperm(len(neg))[:count]]
            dst=torch.cat((pos,neg))
            pairs.append(torch.stack((torch.full_like(dst,source),dst),dim=1))
            remaining-=2*count

        pairs=torch.cat(pairs) if pairs else torch.empty((0,2),dtype=torch.long)
        src,dst=pairs.unbind(dim=1)
        label=TR_label[src,dst].float()
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":torch.full_like(label,float(query_time)),
            "pos_mask":label.bool()
        }

    @staticmethod
    def source_focused_TR_sampling(
            source:int,
            dst:torch.Tensor,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        source를 기준으로 현재 batch event stream의 dst들을 sampling하여 Temporal Reachability 학습 pair를 생성한다.
        
        Class Weight:
            - positive/negative class가 모두 존재하는 경우: 각 class의 loss 기여도가 동일하도록 weight 계산
            - 하나의 class만 존재하는 경우: 모든 sample의 weight를 1로 설정

        Input:
            source
            dst: [B,]
            query_time
            TR_label: [N+1,N+1]
        Return:
            tensor dict:
                src: [B,]
                dst: [B,]
                label: [B,]
                query_t: [B,]
                pos_mask: [B,]
                weight: [B,]
        """
        ### 중복 dst 제거
        dst=torch.unique(dst)

        ### source
        src=torch.full_like(dst,fill_value=source)

        ### TR label
        label=TR_label[src,dst].float()

        ### query time
        query_t=torch.full(
            size=(dst.size(0),),
            fill_value=query_time,
            dtype=torch.float32
        )

        ### Positive / Negative Mask
        pos_mask=label.bool()
        neg_mask=~pos_mask

        ### Class Count
        n_pos=pos_mask.sum().item()
        n_neg=neg_mask.sum().item()
        n_sample=label.numel()

        ### Class Weight
        # 하나의 class만 존재하는 경우 weight=1
        weight=torch.ones_like(label,dtype=torch.float32)

        # 두 class가 모두 존재하는 경우 class balancing
        if n_pos>0 and n_neg>0:
            weight[pos_mask]=n_sample/(2.0*n_pos)
            weight[neg_mask]=n_sample/(2.0*n_neg)

        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":pos_mask,
            "weight":weight
        }

    @staticmethod
    def evaluate_hop_range_TR_sampling(
            n_sample:int,
            source:int,
            query_time:float,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        hop range 1: 1<=hop<3
        hop range 2: 3<=hop<5
        hop range 3: 5<=hop
        """

