import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    dataset_name=kwargs["dataset_name"]
    source=kwargs["source"]
    batch_size=kwargs["batch_size"]
    evaluate_type=kwargs["evaluate_type"]

    ### set test_sample_loader
    dataset_name=kwargs["dataset_name"]
    data=DataUtils.preprocess_snap_dataset(dataset_name=dataset_name)
    graph_df=data["graph_df"]
    graph_dataset=TemporalGraphDataset(df=graph_df)
    graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
    _,_,test_loader=TrainUtils.split_data_loader(data_loader=graph_loader)
    test_TR_result=DataUtils.load_SNAP_TR_result(
        dataset_name=dataset_name,
        batch_size=batch_size,
        purpose="test"
    )
    test_sample_loader=TrainUtils.get_TR_sample_loader(
        source=source,
        data_loader=test_loader,
        TR_result=test_TR_result,
        sampling=evaluate_type
    )

    ### save test_sample_loader
    DataUtils.save_ID_test_sample_loader(
        test_sample_loader=test_sample_loader,
        dataset_name=dataset_name,
        source=source,
        evaluate_type=evaluate_type
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
    parser.add_argument("--source",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=200)
    parser.add_argument("--evaluate_type",choices=["base","hop_range"],default="base")
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "source":args.source,
        "batch_size":args.batch_size,
        "evaluate_type":args.evaluate_type
    }
    main(**app_config)
