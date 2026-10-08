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

    # @staticmethod
    # def source_focused_TR_sampling(
    #         n_sample:int,
    #         source:int,
    #         query_time:float,
    #         TR_label:torch.Tensor,
    #     )->dict[str,torch.Tensor]:
    #     ### positive/negative 목표 sample 개수
    #     n_pos=n_sample//2
    #     n_neg=n_sample-n_pos

    #     ### dst 후보
    #     # padding node(id=0), source 자기 자신 제외
    #     dst_candidates=torch.arange(1,TR_label.shape[1])
    #     dst_candidates=dst_candidates[dst_candidates!=source]

    #     ### positive/negative 후보
    #     source_label=TR_label[source,dst_candidates]
    #     pos_candidates=dst_candidates[source_label]
    #     neg_candidates=dst_candidates[~source_label]

    #     ### 실제 sample 개수
    #     n_pos=min(n_pos,len(pos_candidates))
    #     n_neg=min(n_neg,len(neg_candidates))

    #     ### 중복 없이 random sampling
    #     pos_dst=pos_candidates[torch.randperm(len(pos_candidates))[:n_pos]]
    #     neg_dst=neg_candidates[torch.randperm(len(neg_candidates))[:n_neg]]

    #     ### 실제 총 sample 개수
    #     n_sample=n_pos+n_neg

    #     ### positive -> negative 순서
    #     src=torch.full((n_sample,),source,dtype=torch.long)
    #     dst=torch.cat([ pos_dst,neg_dst])
    #     label=torch.cat([
    #         torch.ones(n_pos,dtype=torch.float32),
    #         torch.zeros(n_neg,dtype=torch.float32)
    #     ])
    #     query_t=torch.full((n_sample,),query_time,dtype=torch.float32)
    #     return {
    #         "src":src,
    #         "dst":dst,
    #         "label":label,
    #         "query_t":query_t,
    #         "pos_mask":label.bool()
    #     }

    @staticmethod
    def TR_sampling_for_evaluate(
            n_sample:int,
            source:int,
            query_time:float,
            TR_label:torch.Tensor,
        ):
        """
        평가용 TR Sampling.

        총 n_sample개의 positive pair + negative pair (1:1비율)를 생성한다.
        source를 기준으로 dst들을 랜덤하게 샘플링한다.
        padding node(id=0), source 자기 자신을 dst 후보에서 제외하며, 동일한 dst는 중복 샘플링하지 않는다.
        
        Input:
            n_sample
            source
            query_time
            TR_label: [N+1,N+1]
        Return:
            tensor dict:
                src: [n_sample,]
                dst: [n_sample,]
                label: [n_sample,]
                query_t: [n_sample,]
        """
        ### positive/negative 목표 sample 개수
        n_pos=n_sample//2
        n_neg=n_sample-n_pos

        ### dst 후보
        # padding node(id=0), source 자기 자신 제외
        dst_candidates=torch.arange(1,TR_label.shape[1])
        dst_candidates=dst_candidates[dst_candidates!=source]

        ### positive/negative 후보
        source_label=TR_label[source,dst_candidates]
        pos_candidates=dst_candidates[source_label]
        neg_candidates=dst_candidates[~source_label]

        ### 실제 sample 개수
        n_pos=min(n_pos,len(pos_candidates))
        n_neg=min(n_neg,len(neg_candidates))

        ### 중복 없이 random sampling
        pos_dst=pos_candidates[torch.randperm(len(pos_candidates))[:n_pos]]
        neg_dst=neg_candidates[torch.randperm(len(neg_candidates))[:n_neg]]

        ### 실제 총 sample 개수
        n_sample=n_pos+n_neg

        ### positive -> negative 순서
        src=torch.full((n_sample,),source,dtype=torch.long)
        dst=torch.cat([ pos_dst,neg_dst])
        label=torch.cat([
            torch.ones(n_pos,dtype=torch.float32),
            torch.zeros(n_neg,dtype=torch.float32)
        ])
        query_t=torch.full((n_sample,),query_time,dtype=torch.float32)
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":label.bool()
        }

    @staticmethod
    def TR_sampling_for_evaluate_hop_range(
            n_sample:int,
            source:int,
            query_time:float,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        평가용 TR Sampling.
        positive node pair중 각 hop range에서 n개씩 sampling하여 총 n_sample개 생성한다.
        
        hop range 1: 1<=hop<3
        hop range 2: 3<=hop<5
        hop range 3: 5<=hop

        padding node(id=0), source 자기 자신을 dst 후보에서 제외하며, 동일한 dst는 중복 샘플링하지 않는다.
        특정 hop range의 후보가 부족한 경우, 해당 range에서는 존재하는 개수만큼만 sampling한다.

        Input:
            n_sample
            source
            query_time
            TR_label: [N+1,N+1]
            TR_hop: [N+1,N+1]
        Return:
            src: [n_sample,]
            dst: [n_sample,]
            label: [n_sample,]
            query_t: [n_sample,]
            range_1_mask: [n_sample,]
            range_2_mask: [n_sample,]
            range_3_mask: [n_sample,]
        """
        ### hop range별 목표 sample 개수
        n_range_1=n_sample//3
        n_range_2=n_sample//3
        n_range_3=n_sample//3

        ### dst 후보
        # padding node(id=0), source 자기 자신 제외
        dst_candidates=torch.arange(1,TR_label.shape[1])
        dst_candidates=dst_candidates[dst_candidates!=source]

        ### positive 후보
        source_label=TR_label[source,dst_candidates]
        pos_candidates=dst_candidates[source_label]

        ### positive 후보의 hop
        pos_hop=TR_hop[source,pos_candidates]

        ### hop range별 후보
        range_1_candidates=pos_candidates[(1<=pos_hop)&(pos_hop<3)]
        range_2_candidates=pos_candidates[(3<=pos_hop)&(pos_hop<5)]
        range_3_candidates=pos_candidates[5<=pos_hop]

        ### 실제 sample 개수
        n_range_1=min(n_range_1,len(range_1_candidates))
        n_range_2=min(n_range_2,len(range_2_candidates))
        n_range_3=min(n_range_3,len(range_3_candidates))

        ### 중복 없이 random sampling
        range_1_dst=range_1_candidates[torch.randperm(len(range_1_candidates))[:n_range_1]]
        range_2_dst=range_2_candidates[torch.randperm(len(range_2_candidates))[:n_range_2]]
        range_3_dst=range_3_candidates[torch.randperm(len(range_3_candidates))[:n_range_3]]

        ### positive -> hop range 순서로 결합
        dst=torch.cat([
            range_1_dst,
            range_2_dst,
            range_3_dst
        ])

        ### 실제 총 sample 개수
        n_sample=len(dst)
        src=torch.full((n_sample,),source,dtype=torch.long)
        label=torch.ones(n_sample,dtype=torch.float32)
        query_t=torch.full((n_sample,),query_time,dtype=torch.float32)

        ### hop range mask
        range_1_mask=torch.zeros(n_sample,dtype=torch.bool)
        range_2_mask=torch.zeros(n_sample,dtype=torch.bool)
        range_3_mask=torch.zeros(n_sample,dtype=torch.bool)
        range_1_mask[:n_range_1]=True
        range_2_mask[n_range_1:n_range_1+n_range_2]=True
        range_3_mask[n_range_1+n_range_2:]=True
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "range_1_mask":range_1_mask,
            "range_2_mask":range_2_mask,
            "range_3_mask":range_3_mask
        }
