import pandas as pd
import numpy as np
from bisect import bisect_right

class TemporalGraph:
    def __init__(self,graph_df:pd.DataFrame):
        self.graph_df=graph_df

        ### graph information
        self.n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
        self.n_event=int(graph_df["idx"].max())
        self.max_u=int(graph_df["u"].max())
        self.max_t=float(graph_df["t"].max())

        ### save in_adj, out_adj info
        self.edge_events=[]

        # in_adj for GNN
        in_adj=[[] for _ in range(self.n_node+1)]
        in_adj_edge=[[] for _ in range(self.n_node+1)]
        in_adj_t=[[] for _ in range(self.n_node+1)]

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

        # edge_events를 t 기준 오름차순 정렬
        self.edge_events.sort(key=lambda x:x[2])
        self.edge_event_times=[event[2] for event in self.edge_events]

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


