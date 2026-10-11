import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from graph import TemporalGraph
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    """
    n_node=100 -> batch_size=32
    n_node=500, 1000 -> batch_size=200 
    """
    graph_type=kwargs["graph_type"]
    purpose=kwargs["purpose"]
    n_node=kwargs["n_node"]
    batch_size=kwargs["batch_size"]

    file_name=f"{graph_type}_{purpose}_N{n_node}"
    graph_df_list=DataUtils.load_graph_df_list_pickle(
        file_name=file_name,
        purpose=purpose
    )

    ### compute TR_result of each graph_df
    TR_result_list=[]
    for graph_df in tqdm(graph_df_list,desc=f"Compute TR result..."):
        graph=TemporalGraph(graph_df=graph_df)
        graph_dataset=TemporalGraphDataset(df=graph_df)
        graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)

        ### compute TR_result
        TR_result=TrainUtils.get_TR_result_using_Time_Centric(
            graph=graph,
            data_loader=graph_loader
        )
        TR_result_list.append(TR_result)

    ### save TR_result list
    file_name=f"{file_name}_B{batch_size}"
    DataUtils.save_TR_result_list_to_pt(
        TR_result_list=TR_result_list,
        file_name=file_name,
        purpose=purpose
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
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
        default="train"
    )
    parser.add_argument("--n_node",
        type=int,
        choices=[100,500,1000],
        default=100
    )
    parser.add_argument("--batch_size",type=int,default=32)
    args=parser.parse_args()
    app_config={
        "graph_type":args.graph_type,
        "purpose":args.purpose,
        "n_node":args.n_node,
        "batch_size":args.batch_size
    }
    main(**app_config)