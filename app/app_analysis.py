import os
import threading
import queue
import networkx as nx
import numpy as np
import argparse
from tqdm import tqdm
from tr_gnn import DataUtils,GraphAnalysis,ModelTrainUtils

def app_analysis(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            check_elements of 7-types graphs
            """
            if config['mode']=="test":
                graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
                for graph_type in graph_type_list:
                    graph_list=DataUtils.load_from_pickle(
                        file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_list",
                        dir_type=f"graph",
                        mode=config['mode'],
                        num_nodes=config['num_nodes'],
                        is_print=False
                    )
                    N_list=[]
                    E_s_list=[]
                    E_list=[]
                    for graph in graph_list:
                        N,E_s,E=GraphAnalysis.check_elements(graph=graph)
                        N_list.append(N)
                        E_s_list.append(E_s)
                        E_list.append(E)
                    print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean num_nodes: {int(np.mean(N_list))}")
                    print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean num_static_edgs MEAN: {np.mean(E_s_list)} MAX: {np.max(E_s_list)} MIN: {np.min(E_s_list)}")
                    print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean num_edge_events MEAN: {np.mean(E_list)} MAX: {np.max(E_list)} MIN: {np.min(E_list)}")
                    print()
            else: # train, val
                graph_list=DataUtils.load_from_pickle(
                    file_name=f"{config['mode']}_{config['num_nodes']}_list",
                    dir_type=f"graph",
                    mode=config['mode'],
                    num_nodes=config['num_nodes'],
                    is_print=False
                )
                N_list=[]
                E_s_list=[]
                E_list=[]
                for graph in graph_list:
                    N,E_s,E=GraphAnalysis.check_elements(graph=graph)
                    N_list.append(N)
                    E_s_list.append(E_s)
                    E_list.append(E)
                print(f"{config['mode']}_{config['num_nodes']} graphs mean num_nodes: {int(np.mean(N_list))}")
                print(f"{config['mode']}_{config['num_nodes']} graphs mean num_static_edgs MEAN: {np.mean(E_s_list)} MAX: {np.max(E_s_list)} MIN: {np.min(E_s_list)}")
                print(f"{config['mode']}_{config['num_nodes']} graphs mean num_edge_events MEAN: {np.mean(E_list)} MAX: {np.max(E_list)} MIN: {np.min(E_list)}")

        case 2:
            """
            App 2.
            check_elements of SNAP datasets
            """
        case 3:
            """
            App 3.
            check_TR_ratio for one source
            """
            if config['mode']=="test":
                print(f"<<Check {config['mode']}_{config['num_nodes']} TR ratio>>")
                graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
                TR_ratio_dict={}
                for graph_type in graph_type_list:
                    all_TR_ratio_list=[]
                    for graph_id in range(5):
                        src_list=DataUtils.load_from_pickle(
                            file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_{graph_id}_src_list",
                            dir_type=f"dataset",
                            mode=config['mode'],
                            num_nodes=config['num_nodes'],
                            is_print=False
                        )
                        TR_ratio_list=[]
                        for src in src_list:
                            traj=DataUtils.load_from_pickle(
                                file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_{graph_id}_traj_{src}",
                                dir_type=f"dataset",
                                mode=config['mode'],
                                num_nodes=config['num_nodes'],
                                is_print=False
                            )
                            TR_ratio=GraphAnalysis.check_tR_ratio(r=traj)
                            TR_ratio_list.append(TR_ratio)
                        all_TR_ratio_list+=TR_ratio_list
                    TR_ratio_dict[graph_type]=np.mean(all_TR_ratio_list)
                for graph_type,TR_ratio in TR_ratio_dict.items():
                    print(f"{config['mode']}_{config['num_nodes']}_{graph_type} mean TR ratio: {TR_ratio}")
            else: # train, val
                print(f"<<Check {config['mode']}_{config['num_nodes']} TR ratio>>")
                all_trajs_list=DataUtils.load_from_pickle(
                    file_name=f"{config['mode']}_{config['num_nodes']}_trajs_list",
                    dir_type=f"dataset",
                    mode=config['mode'],
                    num_nodes=config['num_nodes'],
                    is_print=False
                )
                all_TR_ratio_list=[]
                for trajs in all_trajs_list:
                    TR_ratio_list=[]
                    for traj in trajs:
                        TR_ratio=GraphAnalysis.check_tR_ratio(r=traj)
                        TR_ratio_list.append(TR_ratio)
                    all_TR_ratio_list+=TR_ratio_list
                print(f"{config['mode']}_{config['num_nodes']} mean TR ratio: {np.mean(all_TR_ratio_list)}")

if __name__=="__main__":
    """
    Execute app_analysis
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'mode':args.mode,
        'num_nodes':args.num_nodes
    }
    app_analysis(config=config)