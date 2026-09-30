import argparse
import torch
from utils import DataUtils,TrainUtils

def main(**kwargs):
    """
    """
    dataset_name=kwargs["dataset_name"]
    batch_size=200

    ### load SR, TR result
    train_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        batch_size=batch_size,
        purpose="train"
    )
    val_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        batch_size=batch_size,
        purpose="val"
    )
    test_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        batch_size=batch_size,
        purpose="test"
    )

    ### concat train, val, test TR_label 
    train_TR_label=train_TR_result["label"]
    val_TR_label=val_TR_result["label"]
    test_TR_label=test_TR_result["label"]
    TR_label=torch.cat(
        [
            train_TR_label,
            val_TR_label,
            test_TR_label
        ],
        dim=0
    )

    ### find source candidates
    source_candidates=TrainUtils.get_source_candidates(n_source=10,TR_label=TR_label)
    print(f"{dataset_name} source candidates: {source_candidates}")

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
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name
    }
    main(**app_config)