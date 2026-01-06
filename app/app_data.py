import random
import argparse
from tqdm import tqdm
from tr_gnn import DataUtils,GraphUtils,GraphGenerator

def app_data(config: dict):
    """
    data info:
        train:
            num_graphs (each type): 100 
            num_nodes: 20

        val:
            num_graphs (each type): 5
            num_nodes: 20

        test:
            num_graphs (each type): 5 
            num_nodes: 20, 50, 100, 500, 1000
    """
    match config['app_num']:
        
        case 1:
            """
            App 1. 
            Generate train, val, test graph list and save using pickle.
            """
            if config['mode']=="test":
                graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
                for graph_type in graph_type_list:
                    graph_list=GraphGenerator.generate_7_type_graph_list(num_graphs=config['num_graphs'],graph_type=graph_type,num_nodes=config['num_nodes'],num_times=config['num_time'])
                    DataUtils.save_to_pickle(data=graph_list,file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_list",dir_type="graph",mode=config['mode'],num_nodes=config['num_nodes'])
            else: # train,val
                all_graph_list=[]
                graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
                for graph_type in graph_type_list:
                    graph_list=GraphGenerator.generate_7_type_graph_list(num_graphs=config['num_graphs'],graph_type=graph_type,num_nodes=config['num_nodes'],num_times=config['num_time'])
                    all_graph_list+=graph_list
                DataUtils.save_to_pickle(data=all_graph_list,file_name=f"{config['mode']}_{config['num_nodes']}_list",dir_type="graph",mode=config['mode'],num_nodes=config['num_nodes'])

        case 2:
            """
            App 2. 
            Load SNAP dataset to graph and save using pickle.
            dataset info:
                CollegeMsg
                bitcoin_otc
                bitcoin_alpha
            """
            CollegeMsg=DataUtils.load_SNAP_to_graph(dataset_name='CollegeMsg')
            bitcoin_otc=DataUtils.load_SNAP_to_graph(dataset_name='bitcoin_otc')
            bitcoin_alpha=DataUtils.load_SNAP_to_graph(dataset_name='bitcoin_alpha')

            DataUtils.save_to_pickle(data=CollegeMsg,file_name="CollegeMsg",dir_type="graph",mode="test",is_snap=True)
            DataUtils.save_to_pickle(data=bitcoin_otc,file_name="bitcoin_otc",dir_type="graph",mode="test",is_snap=True)
            DataUtils.save_to_pickle(data=bitcoin_alpha,file_name="bitcoin_alpha",dir_type="graph",mode="test",is_snap=True)

        case 3:
            """
            App 3.
            Convert graph to dataset and save using pickle
                train
                    datastream
                    traj_list
                test
                    src_list
                    datastream
                    traj ...
            """
            if config['mode']=="test":
                graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
                for graph_type in graph_type_list:
                    graph_list=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_list",dir_type=f"graph",mode=config['mode'],num_nodes=config['num_nodes'])
                    for graph_id,graph in enumerate(graph_list):
                        eventstream=GraphUtils.get_eventstream(graph=graph)
                        datastream=GraphUtils.compute_datastream_from_eventstream(eventstream=eventstream,num_nodes=config['num_nodes'])
                        node_list=[i for i in range(config['num_nodes'])]
                        selected_node_list=random.sample(node_list,10) # 랜덤하게 10개의 source node 선택
                        for source_id in selected_node_list:
                            traj=GraphUtils.compute_TR_trajectory_from_eventstream(eventstream=eventstream,num_nodes=config['num_nodes'],source_id=source_id)
                            DataUtils.save_to_pickle(data=traj,file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_{graph_id}_traj_{source_id}",dir_type="dataset",mode=config['mode'],num_nodes=config['num_nodes'])
                        DataUtils.save_to_pickle(data=selected_node_list,file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_{graph_id}_src_list",dir_type="dataset",mode=config['mode'],num_nodes=config['num_nodes'])
                        DataUtils.save_to_pickle(data=datastream,file_name=f"{config['mode']}_{config['num_nodes']}_{graph_type}_{graph_id}_datastream",dir_type="dataset",mode=config['mode'],num_nodes=config['num_nodes'])
            else: # train,val
                all_graph_list=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}_list",dir_type=f"graph",mode=config['mode'],num_nodes=config['num_nodes'])
                all_datastream_list=[]
                all_trajs_list=[]
                for graph in all_graph_list:
                    eventstream=GraphUtils.get_eventstream(graph=graph)
                    datastream=GraphUtils.compute_datastream_from_eventstream(eventstream=eventstream,num_nodes=config['num_nodes'])
                    trajs=[]
                    for source_id in range(config['num_nodes']):
                        traj=GraphUtils.compute_TR_trajectory_from_eventstream(eventstream=eventstream,num_nodes=config['num_nodes'],source_id=source_id)
                        trajs.append(traj)
                    all_datastream_list.append(datastream)
                    all_trajs_list.append(trajs)
                DataUtils.save_to_pickle(data=all_datastream_list,file_name=f"{config['mode']}_{config['num_nodes']}_datastream_list",dir_type="dataset",mode=config['mode'],num_nodes=config['num_nodes'])
                DataUtils.save_to_pickle(data=all_trajs_list,file_name=f"{config['mode']}_{config['num_nodes']}_trajs_list",dir_type="dataset",mode=config['mode'],num_nodes=config['num_nodes'])


if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
    }
    app_data(config=config)