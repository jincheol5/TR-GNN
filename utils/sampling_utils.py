import torch

class SamplingUtils:
    @staticmethod
    def source_independent_TR_sampling(
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
    def source_dependent_TR_sampling(
            source:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        """


















    @staticmethod
    def hop_range_TR_sampling(
            source:int,
            n_sample:int,
            query_time:float,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        ):
        """
        positive pair: 최대 n_sample//2 개
            - hop_range_1: 1 <= hop < 5 -> 30%
            - hop_range_2: 5 <= hop < 10 -> 30%
            - hop_range_3: 10 <= hop -> 40%
            - 각 range에서 후보가 부족한 경우 가능한 수만 sampling
            - 부족분에 대한 추가 sampling은 수행하지 않음

        negative pair:
            - 최종 positive sample 수와 동일한 수를 sampling
            - negative 후보가 부족한 경우 가능한 수만 sampling

        positive dst 순서: [hop_range_1,hop_range_2,hop_range_3]

        Input:
            source
            n_sample
            query_time
            TR_label: [N+1,N+1] bool tensor
            TR_hop: [N+1,N+1] long tensor
        Return:
            src
            dst
            label
            query_t
            pos_mask
            hop_range_1_mask
            hop_range_2_mask
            hop_range_3_mask
            n_hop_range_1
            n_hop_range_2
            n_hop_range_3
        """
        ### positive 목표 sample 수
        n_pos_sample=n_sample//2

        ### positive / negative 후보 생성
        source_label=TR_label[source]
        source_hop=TR_hop[source]
        pos_candidates=torch.where(source_label)[0]
        neg_candidates=torch.where(~source_label)[0]

        # padding node(0), source 자기 자신 제외
        pos_candidates=pos_candidates[
            (pos_candidates!=0)&
            (pos_candidates!=source)
        ]
        neg_candidates=neg_candidates[
            (neg_candidates!=0)&
            (neg_candidates!=source)
        ]

        ### positive 후보를 hop range별로 분리
        pos_hop=source_hop[pos_candidates]
        hop_range_1_candidates=pos_candidates[
            (pos_hop>=1)&(pos_hop<5)
        ]
        hop_range_2_candidates=pos_candidates[
            (pos_hop>=5)&(pos_hop<10)
        ]
        hop_range_3_candidates=pos_candidates[
            pos_hop>=10
        ]

        ### range별 목표 sampling 개수
        n_hop_range_1=int(n_pos_sample*0.3)
        n_hop_range_2=int(n_pos_sample*0.3)
        n_hop_range_3=n_pos_sample-n_hop_range_1-n_hop_range_2

        ### range_1 sampling
        hop_range_1_dst=torch.empty(0,dtype=torch.long)
        if hop_range_1_candidates.numel()>0:
            perm=torch.randperm(hop_range_1_candidates.numel())
            hop_range_1_dst=hop_range_1_candidates[
                perm[:min(n_hop_range_1,hop_range_1_candidates.numel())]
            ]

        ### range_2 sampling
        hop_range_2_dst=torch.empty(0,dtype=torch.long)
        if hop_range_2_candidates.numel()>0:
            perm=torch.randperm(hop_range_2_candidates.numel())
            hop_range_2_dst=hop_range_2_candidates[
                perm[:min(n_hop_range_2,hop_range_2_candidates.numel())]
            ]

        ### range_3 sampling
        hop_range_3_dst=torch.empty(0,dtype=torch.long)
        if hop_range_3_candidates.numel()>0:
            perm=torch.randperm(hop_range_3_candidates.numel())
            hop_range_3_dst=hop_range_3_candidates[
                perm[:min(n_hop_range_3,hop_range_3_candidates.numel())]
            ]

        ### 최종 positive
        # [range_1|range_2|range_3]
        pos_dst=torch.cat([
            hop_range_1_dst,
            hop_range_2_dst,
            hop_range_3_dst
        ])

        ### negative sampling
        # 실제 positive sample 수와 동일하게 sampling
        n_neg_sample=pos_dst.numel()
        neg_dst=torch.empty(0,dtype=torch.long)
        if neg_candidates.numel()>0:
            perm=torch.randperm(neg_candidates.numel())
            neg_dst=neg_candidates[
                perm[:min(n_neg_sample,neg_candidates.numel())]
            ]

        ### output
        # [positive|negative]
        dst=torch.cat([pos_dst,neg_dst])
        label=torch.cat([
            torch.ones(pos_dst.numel(),dtype=torch.float32),
            torch.zeros(neg_dst.numel(),dtype=torch.float32)
        ])

        ### positive mask
        pos_mask=torch.zeros(dst.numel(),dtype=torch.bool)
        pos_mask[:pos_dst.numel()]=True

        ### hop range masks
        hop_range_1_mask=torch.zeros(dst.numel(),dtype=torch.bool)
        hop_range_2_mask=torch.zeros(dst.numel(),dtype=torch.bool)
        hop_range_3_mask=torch.zeros(dst.numel(),dtype=torch.bool)

        # range 1
        hop_range_1_start=0
        hop_range_1_end=hop_range_1_dst.numel()
        hop_range_1_mask[hop_range_1_start:hop_range_1_end]=True

        # range 2
        hop_range_2_start=hop_range_1_end
        hop_range_2_end=hop_range_2_start+hop_range_2_dst.numel()
        hop_range_2_mask[hop_range_2_start:hop_range_2_end]=True

        # range 3
        hop_range_3_start=hop_range_2_end
        hop_range_3_end=hop_range_3_start+hop_range_3_dst.numel()
        hop_range_3_mask[hop_range_3_start:hop_range_3_end]=True

        ### src/query time
        src=torch.full(
            (dst.numel(),),
            source,
            dtype=torch.long
        )
        query_t=torch.full(
            (dst.numel(),),
            query_time,
            dtype=torch.float32
        )
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":pos_mask,
            "hop_range_1_mask":hop_range_1_mask,
            "hop_range_2_mask":hop_range_2_mask,
            "hop_range_3_mask":hop_range_3_mask,
            "n_hop_range_1":hop_range_1_dst.numel(),
            "n_hop_range_2":hop_range_2_dst.numel(),
            "n_hop_range_3":hop_range_3_dst.numel()
        }
