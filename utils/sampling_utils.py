import torch

class SamplingUtils:
    @staticmethod
    def source_independent_TR_sampling(
            sources:list,
            n_pair:int,
            n_sample:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        sources를 랜덤 순회하면서 source별 positive/negative TR pair를 sampling한다.

        [1] 기존 sources sampling
            - sources는 랜덤 순서로 선택한다.
            - 각 source에서 최대 n_pair개의 positive/negative pair를 sampling한다.
            - positive/negative 중 한쪽이 n_pair보다 부족하면 작은 쪽의 수에 맞춘다.
            ex) pos=10, neg=4, n_pair=5 -> pos=4, neg=4
            - positive/negative 중 한쪽이 0개이면 해당 source는 건너뛴다.
            - 전체 sample 수(pos + neg)가 n_sample에 도달하면 종료한다.

        [2] 부족한 sample 보충
            - 기존 sources를 모두 사용했는데 n_sample을 채우지 못한 경우 수행한다.
            - sources에 포함되지 않은 source를 랜덤 순회한다.
            - 현재 positive/negative sample 수를 기준으로 부족한 쪽을 보충한다.
            - 전체 sample 수가 n_sample에 도달하면 종료한다.

        padding node(id=0)와 source 자기 자신은 destination 후보에서 제외한다.
        동일 source에서 sampling되는 destination은 중복되지 않는다.

        Input:
            sources: 우선적으로 사용할 source node list
            n_pair: source 하나에서 positive/negative 각각 최대 sampling할 pair 수
            n_sample: 최종적으로 생성할 전체 positive + negative sample 수
            query_time: query timestamp
            TR_label: [N+1, N+1] bool tensor
        Return:
            dict:
                src: [n_sample,] long tensor
                dst: [n_sample,] long tensor
                label: [n_sample,] float tensor
                query_t: [n_sample,] float tensor
                pos_mask: [n_sample,] bool tensor
        """
        n_node=TR_label.size(0)-1
        sampled_src=[]
        sampled_dst=[]
        sampled_label=[]
        n_pos=0
        n_neg=0

        ### sources random shuffle
        sources=list(map(int, sources))
        perm=torch.randperm(len(sources)).tolist()
        sources=[sources[i] for i in perm]

        ### 기존 sources에서 sampling
        for source in sources:
            remaining=n_sample-(n_pos+n_neg)
            if remaining<2:
                break
            pos_nodes=torch.nonzero(
                TR_label[source],
                as_tuple=False
            ).flatten()
            neg_nodes=torch.nonzero(
                ~TR_label[source],
                as_tuple=False
            ).flatten()

            # padding node, 자기 자신 제외
            pos_nodes=pos_nodes[
                (pos_nodes!=0) & (pos_nodes!=source)
            ]
            neg_nodes=neg_nodes[
                (neg_nodes!=0) & (neg_nodes!=source)
            ]

            # 한쪽이 없으면 skip
            if pos_nodes.numel()==0 or neg_nodes.numel()==0:
                continue

            # pos / neg 동일한 개수 sampling
            n_pair_sample=min(
                n_pair,
                pos_nodes.numel(),
                neg_nodes.numel(),
                remaining//2
            )

            if n_pair_sample==0:
                continue

            # positive sampling
            selected=pos_nodes[torch.randperm(pos_nodes.numel())[:n_pair_sample]]
            sampled_src.append(torch.full_like(selected,source,dtype=torch.long))
            sampled_dst.append(selected.long())
            sampled_label.append(torch.ones_like(selected,dtype=torch.float32))
            n_pos+=n_pair_sample

            # negative sampling
            selected=neg_nodes[torch.randperm(neg_nodes.numel())[:n_pair_sample]]
            sampled_src.append(torch.full_like(selected,source,dtype=torch.long))
            sampled_dst.append(selected.long())
            sampled_label.append(torch.zeros_like(selected,dtype=torch.float32))
            n_neg+=n_pair_sample

        ### 부족한 sample이 있으면 sources에 없던 source에서 보충
        if n_pos+n_neg<n_sample:
            used_sources=set(sources)
            extra_sources=[
                source
                for source in range(1,n_node+1)
                if source not in used_sources
            ]
            perm=torch.randperm(len(extra_sources)).tolist()
            extra_sources=[extra_sources[i] for i in perm]

            # 최종 목표 pos / neg 개수
            target_pos=n_sample//2
            target_neg=n_sample-target_pos

            for source in extra_sources:
                if n_pos+n_neg>=n_sample:
                    break

                pos_nodes=torch.nonzero(TR_label[source],as_tuple=False).flatten()
                neg_nodes=torch.nonzero(~TR_label[source],as_tuple=False).flatten()
                pos_nodes=pos_nodes[
                    (pos_nodes!=0) & (pos_nodes!=source)
                ]
                neg_nodes=neg_nodes[
                    (neg_nodes!=0) & (neg_nodes!=source)
                ]

                ### positive 부족분 보충
                pos_deficit=target_pos-n_pos
                if pos_deficit>0 and pos_nodes.numel()>0:
                    n_pos_sample=min(
                        n_pair,
                        pos_deficit,
                        pos_nodes.numel()
                    )
                    selected=pos_nodes[torch.randperm(pos_nodes.numel())[:n_pos_sample]]
                    sampled_src.append(torch.full_like(selected, source, dtype=torch.long))
                    sampled_dst.append(selected.long())
                    sampled_label.append(torch.ones_like(selected, dtype=torch.float32))
                    n_pos+=n_pos_sample

                ### negative 부족분 보충
                neg_deficit=target_neg-n_neg
                if neg_deficit>0 and neg_nodes.numel()>0:
                    n_neg_sample = min(
                        n_pair,
                        neg_deficit,
                        neg_nodes.numel()
                    )
                    selected=neg_nodes[torch.randperm(neg_nodes.numel())[:n_neg_sample]]
                    sampled_src.append(torch.full_like(selected, source, dtype=torch.long))
                    sampled_dst.append(selected.long())
                    sampled_label.append(torch.zeros_like(selected, dtype=torch.float32))
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
            n_sample:int,
            query_time:float,
            TR_label:torch.Tensor,
            updated_nodes:list[int]|None=None
        )->dict[str,torch.Tensor]:
        """
        source에 대해 총 n_sample개의 positive/negative dst node를 random sampling.
        positive/negative는 동일한 수로 sampling한다.
        한쪽 후보가 부족하면 작은 쪽의 수에 맞춰 sampling한다.
        dst node 후보로 updated_nodes를 먼저 고려한다.
        dst node는 source 자기 자신과 padding node(id=0)는 제외한다.
        sampling된 dst node는 중복되지 않는다.
        부족한 sample에 대한 추가 sampling은 수행하지 않는다.

        Input:
            source: sampling 할 source node
            n_sample: sampling할 전체 pos+neg pair 개수
            query_time: float
            TR_label: [N+1,N+1] bool tensor
            updated_nodes: 해당 시점의 batch eventstream 내에서 업데이트 된 node들

        Return:
            dict:
                src: [n_sample,] long tensor
                dst: [n_sample,] long tensor
                label: [n_sample,] float tensor
                query_t: [n_sample,] float tensor
                pos_mask: [n_sample,] bool tensor
        """
        ### positive/negative 후보 생성
        source_label=TR_label[source]
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

        ### 실제 sampling할 pos/neg 개수
        n_pair=min(
            n_sample//2,
            pos_candidates.numel(),
            neg_candidates.numel()
        )

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
            pos_dst=torch.cat([pos_dst,pos_other[perm[:remain]]])

        ### negative sampling
        neg_updated_mask = torch.tensor(
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
            neg_dst=torch.cat([neg_dst,neg_other[perm[:remain]]])

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
            "pos_mask":pos_mask
        }

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