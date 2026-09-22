import argparse
from torch.utils.data import DataLoader
from graph import TemporalGraph
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    dataset_name=kwargs["dataset_name"]
    purpose=kwargs["purpose"]
    batch_size=kwargs["batch_size"]

    graph_data=DataUtils.preprocess_graph_dataset(dataset_name=dataset_name)
    graph_df=graph_data["graph_df"]
    graph=TemporalGraph(graph_df=graph_df)
    train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)

    train_dataset=TemporalGraphDataset(df=train_df)
    val_dataset=TemporalGraphDataset(df=val_df)
    test_dataset=TemporalGraphDataset(df=test_df)

    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    ### compute TR_result
    match kwargs["purpose"]:
        case "train":
            TR_result=TrainUtils.get_TR_result_using_Time_Centric(
                graph=graph,
                data_loader=train_loader
            )
        case "val":
            TR_result=TrainUtils.get_TR_result_using_Time_Centric(
                graph=graph,
                data_loader=val_loader
            )
        case "test":
            TR_result=TrainUtils.get_TR_result_using_Time_Centric(
                graph=graph,
                data_loader=test_loader
            )

    ### save TR_result
    DataUtils.save_TR_result_to_pt(
        TR_result=TR_result,
        dataset_name=dataset_name,
        purpose=purpose,
        batch_size=batch_size
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()

    args=parser.parse_args()
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "enron",
            "CollegeMsg",
            "bitcoin-alpha",
            "bitcoin-otc"
        ],
        default="ladder"
    )
    parser.add_argument("--purpose",
        type=str,
        choices=["train","val","test"],
        default="ladder"
    )
    parser.add_argument("--batch_size",type=int,default=200)
    app_config={
        "dataset_name":args.dataset_name,
        "purpose":args.purpose,
        "batch_size":args.batch_size
    }
    main(**app_config)