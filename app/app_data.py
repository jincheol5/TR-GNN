import argparse
from trgnn import DataUtils,GraphGenerator

def app_data(config: dict):
    match config['app_num']:
        case 1:
            """
            App 1. 
            Generate train, val, test graph list dictionary and save using pickle.
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
            train_20=GraphGenerator.generate_7_type_graphs(num_graphs=100,num_nodes=20,num_times=config['num_times'])
            val_20=GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20,num_times=config['num_times'])
            test_20=GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=20,num_times=config['num_times'])
            test_50=GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=50,num_times=config['num_times'])
            test_100=GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=100,num_times=config['num_times'])
            test_500=GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=500,num_times=config['num_times'])
            test_1000=GraphGenerator.generate_7_type_graphs(num_graphs=5,num_nodes=1000,num_times=config['num_times'])

            DataUtils.save_to_pickle(data=train_20,file_name="train_20",dir_type="graph")
            DataUtils.save_to_pickle(data=val_20,file_name="val_20",dir_type="graph")
            DataUtils.save_to_pickle(data=test_20,file_name="test_20",dir_type="graph")
            DataUtils.save_to_pickle(data=test_50,file_name="test_50",dir_type="graph")
            DataUtils.save_to_pickle(data=test_100,file_name="test_100",dir_type="graph")
            DataUtils.save_to_pickle(data=test_500,file_name="test_500",dir_type="graph")
            DataUtils.save_to_pickle(data=test_1000,file_name="test_1000",dir_type="graph")

        case 2:
            """
            App 2.
            Convert graph_list_dict to dataset_list
            train_20
            val_20
            test_20
            test_50
            test_100
            test_500
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",dir_type="graph")
            all_graph_list=[]
            for _,graph_list in graph_list_dict.items():
                all_graph_list+=graph_list
            match config['mode']:
                case 'train':
                    DataUtils.save_graph_list_to_dataset_list(
                        graph_list=all_graph_list,
                        num_nodes=config['num_nodes'],
                        dir_type=config['mode']
                    )
                case 'val'|'test':
                    DataUtils.save_graph_list_to_selected_dataset_list(
                        graph_list=all_graph_list,
                        num_nodes=config['num_nodes'],
                        dir_type=config['mode']
                    )

        case 3:
            """
            App 3.
            Convert graph_list to dataset_list
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",dir_type="graph")
            graph_list=graph_list_dict[config['graph_type']]
            match config['mode']:
                case 'train':
                    DataUtils.save_graph_list_to_dataset_list_chunk(
                        graph_list=graph_list,
                        graph_type=config['graph_type'],
                        num_nodes=config['num_nodes'],
                        chunk_size=config['chunk_size'],
                        dir_type=config['mode']
                    )
                case 'val'|'test':
                    DataUtils.save_graph_list_to_selected_dataset_list_chunk(
                        graph_list=graph_list,
                        graph_type=config['graph_type'],
                        num_nodes=config['num_nodes'],
                        chunk_size=config['chunk_size'],
                        dir_type=config['mode']
                    )

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--mode",type=str,default='test')
    parser.add_argument("--num_nodes",type=int,default=20)
    parser.add_argument("--num_times",type=int,default=5)
    parser.add_argument("--graph_type",type=str,default='default')
    parser.add_argument("--chunk_size",type=int,default=1)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'mode':args.mode,
        'num_nodes':args.num_nodes,
        "num_times":args.num_times,
        'graph_type':args.graph_type,
        'chunk_size':args.chunk_size
    }
    app_data(config=config)