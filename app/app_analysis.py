import os
import threading
import queue
import networkx as nx
import numpy as np
import argparse
from tqdm import tqdm
from trgnn import DataUtils,GraphAnalysis,ModelTrainUtils

def app_analysis(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            check_elements
            """
            print(f"<<Check {config['mode']}_{config['num_nodes']} graphs elements>>")
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",dir_type="graph",num_nodes=config['num_nodes'])
            for graph_type,graph_list in graph_list_dict.items():
                N_list=[]
                E_s_list=[]
                E_list=[]
                for graph in graph_list:
                    graph.remove_edges_from(nx.selfloop_edges(graph))
                    N,E_s,E=GraphAnalysis.check_elements(graph=graph)
                    N_list.append(N)
                    E_s_list.append(E_s)
                    E_list.append(E)
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs num_nodes: {np.mean(N_list)}")
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs num_static_edgs mean: {np.mean(E_s_list)} min: {np.min(E_s_list)} max: {np.max(E_s_list)}")
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs num_edge_events mean: {np.mean(E_s_list)} min: {np.min(E_s_list)} max: {np.max(E_s_list)}")
                print()

        case 2:
            """
            App 2.
            check_tR_ratio
                train_50
                val_20
                test_20
                test_50
                test_100
                test_500
            """
            print(f"<<Check {config['mode']}_{config['num_nodes']} graphs tR ratio>>")
            if config['mode']=='train':
                check_size=100 # 1 x 100
            else:
                check_size=50 # 10 x 5
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            dataset_list=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",dir_type=config['mode'],num_nodes=config['num_nodes'])
            dataset_lists=[dataset_list[i:i+check_size] for i in range(0,len(dataset_list),check_size)]
            for graph_type,dataset_list in zip(graph_type_list,dataset_lists):
                all_ratios=[]
                for dataset in dataset_list:
                    label=dataset['label'] # [seq_len,1]
                    tR_ratio=GraphAnalysis.check_tR_ratio(r=label)
                    all_ratios.append(tR_ratio)
                lst=np.array(all_ratios,dtype=float)
                mean_ratio=lst.mean()
                max_ratio=lst.max()
                min_ratio=lst.min()
                print(f"{config['mode']}_{config['num_nodes']} {graph_type} graphs tR_ratio:{mean_ratio} max:{max_ratio} min:{min_ratio}")
                print()

        case 3:
            """
            check_tR_ratio
                test_1000
            """
            print(f"<<Check {config['mode']}_{config['num_nodes']} graphs tR ratio>>")

            chunk_dir_path=os.path.join('..','data','trgnn','test',f"{config['num_nodes']}")
            chunk_files=sorted(
                [f for f in os.listdir(chunk_dir_path) if f.startswith(f"test_{config['num_nodes']}_chunk_{config['chunk_size']}_")],
                key=lambda x: int(x.split("_")[-1].split(".")[0])  # 마지막 index 숫자로 정렬
            )
            chunk_paths=[os.path.join(chunk_dir_path,f) for f in chunk_files]
            num_chunks=len(chunk_paths) # for tqdm

            buffer_queue=queue.Queue(maxsize=2)
            loader_thread=threading.Thread(
                target=ModelTrainUtils.chunk_loader_worker,
                args=(chunk_paths,buffer_queue)
            )
            loader_thread.start()

            label_list=[]
            pbar=tqdm(total=num_chunks,desc="Evaluating chunks...") # tqdm: 전체 chunk 수 기준
            while True:
                dataset_list=buffer_queue.get()
                if dataset_list is None:
                    break
                for dataset in dataset_list:
                    label=dataset['label'] # [seq_len,1]
                    label_list.append(label)
                pbar.update(1) # chunk 처리 완료 → tqdm 1 증가
                del dataset_list # 메모리 정리
            
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']

            check_size=50
            label_lists=[label_list[i:i+check_size] for i in range(0,len(label_list),check_size)]
            for graph_type,sub_label_list in zip(graph_type_list,label_lists):
                all_ratios=[]
                for label in sub_label_list:
                    tR_ratio=GraphAnalysis.check_tR_ratio(r=label)
                    all_ratios.append(tR_ratio)
                lst=np.array(all_ratios,dtype=float)
                mean_ratio=lst.mean()
                max_ratio=lst.max()
                min_ratio=lst.min()
                print(f"{config['mode']}_{config['num_nodes']} {graph_type} graphs tR_ratio:{mean_ratio} max:{max_ratio} min:{min_ratio}")
                print()

if __name__=="__main__":
    """
    Execute app_analysis
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--graph_type",type=str,default="ladder")
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--num_nodes",type=int,default=20)
    parser.add_argument("--chunk_size",type=int,default=10)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        "graph_type":args.graph_type,
        "mode":args.mode,
        "num_nodes":args.num_nodes,
        "chunk_size":args.chunk_size
    }
    app_analysis(config=config)