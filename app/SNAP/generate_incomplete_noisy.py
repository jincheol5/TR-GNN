import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    """
    """
    dataset_name=kwargs["dataset_name"]
    batch_size=kwargs["batch_size"]

    ### load test_loader
    graph_data=DataUtils.preprocess_snap_dataset(dataset_name=dataset_name)
    graph_df=graph_data["graph_df"]
    graph_dataset=TemporalGraphDataset(df=graph_df)
    graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
    _,_,test_loader=TrainUtils.split_data_loader(data_loader=graph_loader)

    ### convert to incomplete and noisy eventsream
    converted_eventstream=TrainUtils.convert_to_incomplete_noisy_eventstream(graph_loader=test_loader)

    ### save converted_eventstream
    DataUtils.save_SNAP_incomplete_noisy_eventstream(
        eventstream=converted_eventstream,
        dataset_name=dataset_name,
        batch_size=batch_size
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "bitcoin-alpha",
            "bitcoin-otc"
        ],
        default="CollegeMsg"
    )
    parser.add_argument("--batch_size",type=int,default=200)
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "batch_size":args.batch_size
    }
    main(**app_config)