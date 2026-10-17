import argparse
from utils import DataUtils,TrainUtils

def main(**kwargs):
    """
    n_node=100 -> batch_size=32
    n_node=500, 1000 -> batch_size=200 
    """
    n_node=kwargs["n_node"]
    batch_size=kwargs["batch_size"]
    purpose=kwargs["purpose"]

    ### set sample_loader_list
    evaluate_data=TrainUtils.get_data_for_evaluate_OOD(
        n_node=n_node,
        batch_size=batch_size,
        purpose=purpose
    )
    evaluate_sample_loader_list=evaluate_data["sample_loader_list"]

    ### save sample_loader_list
    file_name=f"{purpose}_sample_loader_list_N{n_node}_B{batch_size}"
    DataUtils.save_evaluate_sample_loader_list(
        sample_loader_list=evaluate_sample_loader_list,
        file_name=file_name,
        purpose=purpose
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--n_node",
        type=int,
        choices=[100,500,1000],
        default=100
    )
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--purpose",
        type=str,
        choices=["val","test"],
        default="test"
    )
    args=parser.parse_args()
    app_config={
        "n_node":args.n_node,
        "batch_size":args.batch_size,
        "purpose":args.purpose
    }
    main(**app_config)