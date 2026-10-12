import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,ModelUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer,ReaCH_TGN,TR_GNN
from model_train import ModelTrainer,ReaCH_TGN_Trainer,TR_GNN_Trainer

def main(**kwargs):
    ### seed
    seed=kwargs["seed"]
    TrainUtils.set_seed(seed=seed)

    ### 모델 관련 파라미터
    model_name=kwargs["model_name"]
    n_node=100
    node_dim=32
    edge_dim=32
    time_dim=32
    latent_dim=128
    msg_dim=32
    mem_dim=32
    embed_dim=32
    n_layer=1
    n_neighbor=10
    n_head=4

    ### TGN 하이퍼 파라미터
    msg_fn=f"mlp"
    aggr_fn=f"last"

    ### DyGFormer 하이퍼 파라미터
    co_dim=32
    common_dim=32
    max_history_len=10
    patch_size=5

    ### 학습 관련 파라미터
    optimizer=f"adam"
    epoch=100
    early_stop=True
    patience=10
    batch_size=32
    lr=kwargs["lr"]
    sampling=kwargs["sampling"]
    source=kwargs["source"]

    ### set model_config
    model_config={
        # 모델 공동 하이퍼 파라미터
        "model_name":kwargs["model_name"],
        "time_dim":time_dim,
        "latent_dim":latent_dim,
        "embed_dim":embed_dim,
        "n_layer":n_layer,
        "n_neighbor":n_neighbor,
        "n_head":n_head,

        # DyGFormer 하이퍼 파라미터
        "co_dim":co_dim,
        "common_dim":common_dim,
        "max_history_len":max_history_len,
        "patch_size":patch_size,
        
        # 학습 관련 파라미터
        "seed":seed,
        "optimizer":optimizer,
        "epoch":epoch,
        "early_stop":early_stop,
        "patience":patience,
        "batch_size":batch_size,
        "lr":lr,

        # sampling 관련 파라미터
        "sampling":sampling,
        "source":source
    }

    ### set dataset
    graph_type_list=[
        "ladder",
        "grid",
        "tree",
        "erdos_renyi",
        "barabasi_albert",
        "community",
        "caveman"
    ]
    train_graph_df_list_dict={}
    for graph_type in graph_type_list:
        file_name=f"{graph_type}_train_N{n_node}"
        graph_df_list=DataUtils.load_graph_df_list_pickle(
            file_name=file_name,
            purpose=f"train"
        )
        train_graph_df_list_dict[graph_type]=graph_df_list


    











    ### set model
    match model_name:
        case "TGAT":
            model=TGAT(
                n_node=n_node,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                graph=None,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
        case "TGN":
            model=TGN(
                n_node=n_node,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                graph=None,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
        case "DyGFormer":
            model=DyGFormer(
                n_node=n_node,
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                graph=None,
                n_layer=n_layer,
                n_head=n_head,
                max_history_len=max_history_len,
                patch_size=patch_size
            )
        case "ReaCH-TGN":
            model=ReaCH_TGN(
                n_node=n_node,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                graph=None,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
        case "TR-GNN":
            model=TR_GNN(
                n_node=n_node,
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                graph=None,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )


if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer","ReaCH-TGN","TR-GNN"],
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0005,0.0001],default=0.0005)
    parser.add_argument("--sampling",
        type=str,
        choices=[
            "focused"
        ],
        default="focused"
    )
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
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "sampling":args.sampling,
        "dataset_name":args.dataset_name,
        "source":args.source,
        "save_model":args.save_model
    }
    main(**app_config)