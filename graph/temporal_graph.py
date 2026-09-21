import random
import pandas as pd
import numpy as np

class TemporalGraph:
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        self.graph_df=graph_df

        ### graph information
        self.n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
        self.n_event=int(graph_df["idx"].max())
        self.bipartite=bipartite
        self.max_u=int(graph_df["u"].max())
        self.max_t=float(graph_df["t"].max())

        ### save in_adj, out_adj info
        self.edge_events=[]

        # in_adj for GNN
        in_adj=[[] for _ in range(self.n_node+1)]
        in_adj_edge=[[] for _ in range(self.n_node+1)]
        in_adj_t=[[] for _ in range(self.n_node+1)]

        # out_adj for algorithm
        out_adj=[[] for _ in range(self.n_node+1)]
        out_adj_edge=[[] for _ in range(self.n_node+1)]
        out_adj_t=[[] for _ in range(self.n_node+1)]
        for event in graph_df.itertuples(index=False): # col: [u,i,t,idx=edge_id]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.t)
            edge_id=int(event.idx)

            # edge event 저장
            self.edge_events.append((src,dst,t,edge_id))

            # incoming adj 저장
            in_adj[dst].append(src)
            in_adj_edge[dst].append(edge_id)
            in_adj_t[dst].append(t)

            # outgoing adj 저장
            out_adj[src].append(dst)
            out_adj_edge[src].append(edge_id)
            out_adj_t[src].append(t)

        ### convert list -> numpy array
        # incoming adj
        self.in_adj=[
            np.asarray(values,dtype=np.int64)
            for values in in_adj
        ]
        self.in_adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in in_adj_edge
        ]
        self.in_adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in in_adj_t
        ]

        # outgoing adj
        self.out_adj=[
            np.asarray(values,dtype=np.int64)
            for values in out_adj
        ]
        self.out_adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in out_adj_edge
        ]
        self.out_adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in out_adj_t
        ]

    def set_random_seed(self,
            seed:int
        ):
        self.rng=random.Random(seed)

    def get_num_node(self):
        return self.n_node

    def get_num_event(self):
        return self.n_event

    def compute_TR(self,
            source:int,
            query_time:float|None=None,
            max_hop:int|None=None
        ):
        """
        Compute Temporal Reachability
        - Min hop
        """
        if max_hop is None:
            max_hop=self.n_node-1

        INF=float("inf")
        NEG_INF=float("-inf")

        TR_info={
            node:{
                "r":0,
                "hop":INF,
                "first_t":INF,
                "last_t":INF
            }
            for node in range(1,self.n_node+1)
        }
        TR_info[source]={
            "r":1,
            "hop":0,
            "first_t":0.0,
            "last_t":NEG_INF
        }

        # 탐색용 상태: node -> (arrival_t, first_t)
        current={
            source:(NEG_INF,0.0)
        }

        # 지금까지 각 node에 가장 일찍 도착한 시간
        best_arrival={
            source:NEG_INF
        }

        for hop in range(1,max_hop+1):
            next_state={}

            for node,(arrival_t,first_t) in current.items():
                times=self.out_adj_t[node]
                neighbors=self.out_adj[node]

                start=np.searchsorted(
                    times,
                    arrival_t,
                    side="right"
                )
                end=(
                    len(times)
                    if query_time is None
                    else np.searchsorted(
                        times,
                        query_time,
                        side="right"
                    )
                )

                for idx in range(start,end):
                    dst=int(neighbors[idx])
                    t=float(times[idx])

                    # 이전 hop에서 이미 더 일찍 도착했으면
                    # 현재 경로는 탐색 가치 없음
                    if best_arrival.get(dst,INF)<=t:
                        continue

                    # 같은 hop에서 이미 더 일찍 도착했으면 제거
                    if dst in next_state and next_state[dst][0]<=t:
                        continue

                    next_state[dst]=(
                        t,
                        t if hop==1 else first_t
                    )

            if not next_state:
                break

            for node,(arrival_t,first_t) in next_state.items():
                # 탐색용 earliest arrival은 계속 갱신
                best_arrival[node]=arrival_t

                # 결과는 최초 발견 때만 저장
                # hop-layer BFS이므로 최초 발견 hop = minimum hop
                if TR_info[node]["r"]==0:
                    TR_info[node]={
                        "r":1,
                        "hop":hop,
                        "first_t":first_t,
                        "last_t":arrival_t
                    }
            current=next_state
        return TR_info
