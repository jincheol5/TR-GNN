import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from graph import TemporalGraph
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    """
    """
    graph_type=kwargs["graph_type"]
    purpose=kwargs["purpose"]
    n_node=kwargs["n_node"]
    batch_size=kwargs["batch_size"]

    graph_df_list_file_name=f"{graph_type}_{purpose}_N{n_node}"
    graph_df_list=DataUtils.load_graph_df_list_pickle(
        file_name=graph_df_list_file_name,
        purpose=purpose
    )

    TR_result_list=[]
    for graph_df in tqdm(
            graph_df_list,
            desc=f"Compute TR_result of each {graph_type}_{purpose}_N{n_node}_B{batch_size} graph_df..."
        ):
        graph=TemporalGraph(graph_df=graph_df)
        graph_dataset=TemporalGraphDataset(df=graph_df)
        graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
        TR_result=TrainUtils.get_TR_result(graph=graph,data_loader=graph_loader)
        TR_result_list.append(TR_result)

    TR_result_list_file_name=f"{graph_type}_{purpose}_N{n_node}_B{batch_size}"
    DataUtils.save_TR_result_list_to_pt(
        TR_result_list=TR_result_list,
        file_name=TR_result_list_file_name,
        purpose=purpose
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()

    args=parser.parse_args()
    parser.add_argument("--graph_type",
        type=str,
        choices=[
            "ladder",
            "grid",
            "tree",
            "erdos_renyi",
            "barabasi_albert",
            "community",
            "caveman"
        ],
        default="ladder"
    )
    parser.add_argument("--purpose",
        type=str,
        choices=["train","val","test"],
        default="ladder"
    )
    parser.add_argument("--n_node",
        type=int,
        choices=[100,500,1000],
        default=100
    )
    parser.add_argument("--batch_size",type=int,default=200)
    app_config={
        "graph_type":args.graph_type,
        "purpose":args.purpose,
        "n_node":args.n_node,
        "batch_size":args.batch_size
    }
    main(**app_config)