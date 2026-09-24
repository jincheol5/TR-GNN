import torch

class SamplingUtils:
    @staticmethod
    def source_independent_TR_sampling(
            sources:list,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        각 source 마다 n_pair 개의 positive/negative dst node를 random sampling.
        sources는 중복이 없도록 한다.
        dst node는 src node 자기자신과 padding node (id=0)는 제외한다.
        각 source에 대해 sampling 된 dst node는 중복이 없도록 한다. 
        pair 수가 비균형 할 경우에도 부족한 대로 수를 유지한다. (ex: pos_pair:10, neg_pair:5)
        한 쪽 밖에 없는 경우에도 유효한 쪽만 유지하여 sampling 한다. (ex: pos_pair:0, neg_pair:10)
        source 별 sampling이 끝난 후, positive/negative 별로 sample 개수(n_pair x len(sources))가 부족한 경우 다음 과정 수행:
        - sources에 없던 source를 랜덤 선정
        - 부족한 positive/negative sample 수가 n_pair 이상인 경우, n_pair 수 만큼 random TR Sampling 수행. 
        - 부족한 positive/negative sample 수가 n_pair 미만인 경우, 부족한 수 만큼 random TR Sampling 수행.
        - 부족한 sample 수 채워질 때 까지 반복.

        Input:
            sources: sampling 할 source node list
            n_pair: source 당 sampling할 pos/neg pair node 개수
            query_time: float
            TR_label: [N+1,N+1] bool tensor, reachable하면 True, unreachable하면 False
        Return:
            Return: dict
                src: [n_sample,] long tensor
                dst: [n_sample,] long tensor
                label: [n_sample,] float tensor (1.0 or 0.0)
                query_t: [n_sample,] float tensor
                pos_mask: [n_sample,] bool tensor
        """
        n_node=TR_label.size(0)-1
        n_sample=n_pair*len(sources)

        sampled_src=[]
        sampled_dst=[]
        sampled_label=[]

        n_pos=0
        n_neg=0

        ### 기존 sources에서 sampling
        for source in sources:
            source=int(source)

            pos_nodes=torch.nonzero(TR_label[source],as_tuple=False).flatten()
            neg_nodes=torch.nonzero(~TR_label[source],as_tuple=False).flatten()

            # padding node, 자기 자신 제외
            pos_nodes=pos_nodes[
                (pos_nodes!=0)&(pos_nodes!=source)
            ]
            neg_nodes=neg_nodes[
                (neg_nodes!=0)&(neg_nodes!=source)
            ]

            # positive sampling
            n_pos_sample=min(n_pair,pos_nodes.numel())
            if n_pos_sample>0:
                selected=pos_nodes[
                    torch.randperm( # 0 ~ len(pos_nodes)-1의 index를 무작위 순서로 생성
                        pos_nodes.numel()
                    )[:n_pos_sample] 
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.ones_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_pos+=n_pos_sample

            # negative sampling
            n_neg_sample=min(n_pair,neg_nodes.numel())
            if n_neg_sample>0:
                selected=neg_nodes[
                    torch.randperm(
                        neg_nodes.numel()
                    )[:n_neg_sample]
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.zeros_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_neg+=n_neg_sample

        ### 기존 sources에 포함되지 않은 source 후보
        used_sources=set(map(int,sources))
        extra_sources=[
            source
            for source in range(1,n_node+1)
            if source not in used_sources
        ]
        if extra_sources:
            perm=torch.randperm(len(extra_sources)).tolist()
            extra_sources=[
                extra_sources[i]
                for i in perm
            ]

        ### 부족한 positive / negative sample 보충
        for source in extra_sources:
            pos_deficit=max(0,n_sample-n_pos)
            neg_deficit=max(0,n_sample-n_neg)
            if pos_deficit==0 and neg_deficit==0:
                break

            pos_nodes=torch.nonzero(
                TR_label[source],
                as_tuple=False
            ).flatten()

            neg_nodes=torch.nonzero(
                ~TR_label[source],
                as_tuple=False
            ).flatten()

            pos_nodes=pos_nodes[
                (pos_nodes!=0)&(pos_nodes!=source)
            ]
            neg_nodes=neg_nodes[
                (neg_nodes!=0)&(neg_nodes!=source)
            ]

            # positive 보충
            n_pos_sample=min(n_pair,pos_deficit,pos_nodes.numel())
            if n_pos_sample>0:
                selected=pos_nodes[
                    torch.randperm(
                        pos_nodes.numel()
                    )[:n_pos_sample]
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.ones_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_pos+=n_pos_sample

            # negative 보충
            n_neg_sample=min(n_pair,neg_deficit,neg_nodes.numel())
            if n_neg_sample>0:
                selected=neg_nodes[
                    torch.randperm(
                        neg_nodes.numel()
                    )[:n_neg_sample]
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.zeros_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_neg+=n_neg_sample

        ### 결과 tensor 생성
        src=torch.cat(sampled_src)
        dst=torch.cat(sampled_dst)
        label=torch.cat(sampled_label)
        query_t=torch.full_like(
            label,
            float(query_time),
            dtype=torch.float32
        )
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":label.bool()
        }

    @staticmethod
    def source_dependent_TR_sampling(
            source:int,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor,
            updated_nodes:list[int]|None=None
        )->dict[str,torch.Tensor]:
        """
        source에 대해 n_pair 개의 positive/negative dst node를 random sampling.
        dst node 후보로 updated_nodes 먼저 고려한다.
        dst node는 source node 자기자신과 padding node (id=0)는 제외한다.
        sampling 된 dst node는 중복이 없도록 한다. 
        
        Input:
            source: sampling 할 source node
            n_pair: sampling할 pos/neg pair node 개수
            query_time: float
            TR_label: [N+1,N+1] bool tensor, reachable하면 True, unreachable하면 False
            updated_nodes: 해당 시점의 batch eventstream 내에서 업데이트 된 node들
        Return:
            Return: dict
                src: [n_sample,] long tensor
                dst: [n_sample,] long tensor
                label: [n_sample,] float tensor (1.0 or 0.0)
                query_t: [n_sample,] float tensor
                pos_mask: [n_sample,] bool tensor
        """
        ### positive/negative 후보 생성
        source_label=TR_label[source] # [N+1]
        pos_candidates=torch.where(source_label)[0]
        neg_candidates=torch.where(~source_label)[0]

        # padding node(0), source 자기 자신 제외
        pos_candidates=pos_candidates[
            (pos_candidates!=0) &
            (pos_candidates!=source)
        ]
        neg_candidates=neg_candidates[
            (neg_candidates!=0) &
            (neg_candidates!=source)
        ]

        ### updated_nodes
        updated_set=(
            set(updated_nodes)
            if updated_nodes is not None
            else set()
        )
        updated_set.discard(0)
        updated_set.discard(source)

        ### positive sampling
        pos_updated_mask=torch.tensor(
            [int(n) in updated_set for n in pos_candidates.tolist()],
            dtype=torch.bool
        )
        pos_priority=pos_candidates[pos_updated_mask]
        pos_other=pos_candidates[~pos_updated_mask]
        pos_dst=torch.empty(0,dtype=torch.long)

        if pos_priority.numel()>0:
            perm=torch.randperm(pos_priority.numel())
            pos_dst=pos_priority[perm[:min(n_pair,pos_priority.numel())]]

        remain=n_pair-pos_dst.numel()
        if remain>0 and pos_other.numel()>0:
            perm=torch.randperm(pos_other.numel())
            pos_dst=torch.cat([
                pos_dst,
                pos_other[perm[:min(remain,pos_other.numel())]]
            ])

        ### negative sampling
        neg_updated_mask=torch.tensor(
            [int(n) in updated_set for n in neg_candidates.tolist()],
            dtype=torch.bool
        )
        neg_priority=neg_candidates[neg_updated_mask]
        neg_other=neg_candidates[~neg_updated_mask]
        neg_dst=torch.empty(0,dtype=torch.long)
        if neg_priority.numel()>0:
            perm=torch.randperm(neg_priority.numel())
            neg_dst=neg_priority[perm[:min(n_pair,neg_priority.numel())]]

        remain=n_pair-neg_dst.numel()
        if remain>0 and neg_other.numel()>0:
            perm=torch.randperm(neg_other.numel())
            neg_dst=torch.cat([
                neg_dst,
                neg_other[perm[:min(remain, neg_other.numel())]]
            ])

        ### output
        dst=torch.cat([pos_dst,neg_dst])
        label=torch.cat([
            torch.ones(pos_dst.numel(),dtype=torch.float32),
            torch.zeros(neg_dst.numel(),dtype=torch.float32)
        ])
        pos_mask=torch.cat([
            torch.ones(pos_dst.numel(),dtype=torch.bool),
            torch.zeros(neg_dst.numel(),dtype=torch.bool)
        ])
        src=torch.full((dst.numel(),),source,dtype=torch.long)
        query_t=torch.full((dst.numel(),),query_time,dtype=torch.float32)
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":pos_mask
        }

    @staticmethod
    def hop_range_TR_sampling(
            source:int,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        ):
        """
        positive pair: n_pair 개 
            - hop_range_1: 1 <= hop < 5 -> 30%
            - hop_range_2: 5 <= hop < 10 -> 30%
            - hop_range_3: 10 <= hop -> 40%
            - 각 range에서 부족한 수는 마지막에 남은 positive 후보로 채움
        positive dst 순서: [hop_range_1, hop_range_2, hop_range_3, extra]

        negative pair: n_pair 개 

        Input:
            source
            n_pair
            query_time
            TR_label: [N+1,N+1] bool tensor, reachable하면 True, unreachable하면 False
            TR_hop: [N+1,N+1] long tensor, reachable하면 shortest hop 값, unreachable하면 0
        Return: dict
            src: [n_sample,] long tensor
            dst: [n_sample,] long tensor
            label: [n_sample,] float tensor (1.0 or 0.0)
            query_t: [n_sample,] float tensor
            pos_mask: [n_sample,] bool tensor
            hop_range_1_mask: [n_sample,] bool tensor
            hop_range_2_mask: [n_sample,] bool tensor
            hop_range_3_mask: [n_sample,] bool tensor
        """
        ### positive / negative 후보 생성
        source_label=TR_label[source]
        source_hop=TR_hop[source]
        pos_candidates=torch.where(source_label)[0]
        neg_candidates=torch.where(~source_label)[0]
        # padding node(0), source 자기 자신 제외
        pos_candidates=pos_candidates[
            (pos_candidates!=0) &
            (pos_candidates!=source)
        ]

        ### positive 후보를 hop range별로 분리
        pos_hop=source_hop[pos_candidates]
        hop_range_1_candidates=pos_candidates[
            (pos_hop>=1) &
            (pos_hop<5)
        ]
        hop_range_2_candidates=pos_candidates[
            (pos_hop>=5) &
            (pos_hop<10)
        ]
        hop_range_3_candidates=pos_candidates[
            pos_hop>=10
        ]

        ### range별 sampling 개수
        n_hop_range_1=int(n_pair*0.3)
        n_hop_range_2=int(n_pair*0.3)
        n_hop_range_3=n_pair-n_hop_range_1-n_hop_range_2

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
        hop_range_3_dst=torch.empty(0,dtype=torch.long,)
        if hop_range_3_candidates.numel()>0:
            perm=torch.randperm(hop_range_3_candidates.numel())
            hop_range_3_dst=hop_range_3_candidates[
                perm[:min(n_hop_range_3,hop_range_3_candidates.numel())]
            ]

        ### 우선 range_1 -> range_2 -> range_3 순으로 구성
        hop_range_dst=torch.cat([
            hop_range_1_dst,
            hop_range_2_dst,
            hop_range_3_dst
        ])

        ### 부족한 positive pair를 마지막에 추가
        remain=n_pair-hop_range_dst.numel()
        extra_dst=torch.empty(0,dtype=torch.long,)
        if remain>0:
            sampled_mask=torch.isin(pos_candidates,hop_range_dst)
            remaining_candidates=pos_candidates[~sampled_mask]
            if remaining_candidates.numel()>0:
                perm=torch.randperm(remaining_candidates.numel())
                extra_dst=remaining_candidates[
                    perm[:min(remain,remaining_candidates.numel())]
                ]

        ### 최종 positive
        # [range_1 | range_2 | range_3 | extra]
        pos_dst=torch.cat([
            hop_range_1_dst,
            hop_range_2_dst,
            hop_range_3_dst,
            extra_dst
        ])

        ### negative sampling
        neg_dst=torch.empty(0,dtype=torch.long)
        if neg_candidates.numel()>0:
            perm=torch.randperm(neg_candidates.numel())
            neg_dst=neg_candidates[
                perm[:min(n_pair,neg_candidates.numel())]
            ]

        ### output
        # [positive | negative]
        dst=torch.cat([pos_dst,neg_dst])
        label=torch.cat([
            torch.ones(
                pos_dst.numel(),
                dtype=torch.float32
            ),
            torch.zeros(
                neg_dst.numel(),
                dtype=torch.float32
            )
        ])

        ### positive mask
        pos_mask=torch.zeros(
            dst.numel(),
            dtype=torch.bool
        )
        pos_mask[:pos_dst.numel()]=True

        ### range mask
        hop_range_1_mask=torch.zeros(
            dst.numel(),
            dtype=torch.bool
        )
        hop_range_2_mask=torch.zeros(
            dst.numel(),
            dtype=torch.bool
        )
        hop_range_3_mask=torch.zeros(
            dst.numel(),
            dtype=torch.bool
        )
        # range_1
        hop_range_1_start=0
        hop_range_1_end=hop_range_1_dst.numel()
        hop_range_1_mask[hop_range_1_start:hop_range_1_end]=True

        # range_2
        hop_range_2_start=hop_range_1_end
        hop_range_2_end=hop_range_2_start+hop_range_2_dst.numel()
        hop_range_2_mask[hop_range_2_start:hop_range_2_end]=True

        # range_3
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
            "hop_range_3_mask":hop_range_3_mask
        }