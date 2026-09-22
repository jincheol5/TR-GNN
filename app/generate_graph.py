from tqdm import tqdm
from utils import GraphGenerator,DataUtils,GraphUtils

def main():
    """
    graph info:
        train:
            n_graph (each type): 100 
            n_node: 100

        val:
            n_graph (each type): 5 
            n_node: 100

        test:
            n_graph (each type): 5 
            n_node: 100, 500, 1000
    """
    graph_list_dict_train_100=GraphGenerator.generate_7_type_graphs(n_graph=100,n_node=100)
    graph_list_dict_val_100=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=100)
    graph_list_dict_test_100=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=100)
    graph_list_dict_test_500=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=500)
    graph_list_dict_test_1000=GraphGenerator.generate_7_type_graphs(n_graph=5,n_node=1000)

    all_graph_list_dict={
        "train_N100":graph_list_dict_train_100,
        "val_N100":graph_list_dict_val_100,
        "test_N100":graph_list_dict_test_100,
        "test_N500":graph_list_dict_test_500,
        "test_N1000":graph_list_dict_test_1000
    }

    for key,graph_list_dict in tqdm(
            all_graph_list_dict.items(),
            total=len(all_graph_list_dict),
            desc=f"Convert to graph_df and Save..."
        ):
        purpose,_=key.split("_")
        graph_df_list_dict={
            "ladder":[],
            "grid":[],
            "tree":[],
            "erdos_renyi":[],
            "barabasi_albert":[],
            "community":[],
            "caveman":[]
        }
        # convert to graph_df
        for graph_type,graph_list in graph_list_dict.items():
            for graph in graph_list:
                graph_df=GraphUtils.convert_nx_graph_to_df(graph=graph)
                graph_df_list_dict[graph_type].append(graph_df)

        # save file
        for graph_type,graph_df_list in graph_df_list_dict.items():
                DataUtils.save_graph_df_list_to_pickle(
                    graph_df_list=graph_df_list,
                    file_name=f"{graph_type}_{key}",
                    purpose=purpose
                )

if __name__=="__main__":
    """
    Execute app
    """
    main()