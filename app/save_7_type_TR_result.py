import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from graph import TemporalGraph
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    """
    """

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