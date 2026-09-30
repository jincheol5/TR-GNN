import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer
from model_train import ModelTrainer

def main(**kwargs):
    ### seed
    seed=kwargs["seed"]
    TrainUtils.set_seed(seed=seed)

    ### 모델 관련 파라미터
    model_name=kwargs["model_name"]
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

    ### DyGFormer 하이퍼 파라미터
    co_dim=32
    common_dim=32
    max_history_len=10
    patch_size=5

    ### ReaCH-TGN 하이퍼 파라미터


    ### 학습 관련 파라미터
    optimizer=f"adam"
    epoch=100
    early_stop=True
    patience=10
    batch_size=200
    lr=kwargs["lr"]
    sampling=kwargs["sampling"]
    n_sample=1000
    n_pair=10
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
        "n_sample":n_sample,
        "n_pair":n_pair,
        "source":source
    }

    ### set dataset
    dataset_name=kwargs["dataset_name"]
    data=DataUtils.preprocess_graph_dataset(dataset_name=dataset_name)
    graph_df=data["graph_df"]
    if model_name=="DyGFormer":
        graph=DyGFormer_Graph(
            graph_df=graph_df,
            node_dim=node_dim,
            edge_dim=edge_dim
        )
    else: # TGAT, TGN, ReaCH-TGN
        graph=TGN_Graph(
            graph_df=graph_df,
            node_dim=node_dim,
            edge_dim=edge_dim
        )
    graph.set_random_seed(seed=seed)

    ### set model
    match model_name:
        case "TGAT":
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
        case "TGN":
            model=TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=f"mlp",
                aggr_fn=f"last"
            )
        case "DyGFormer":
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_head=n_head,
                max_history_len=max_history_len,
                patch_size=patch_size
            )

    ### set data_loader
    train_df,val_df,_=TrainUtils.split_graph_df(df=graph_df)
    train_dataset=TemporalGraphDataset(df=train_df)
    val_dataset=TemporalGraphDataset(df=val_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

    ### load TR result
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

    ### set val_sample_list using TR sampling
    val_sample_list=TrainUtils.get_TR_sample_list(
        data_loader=val_loader,
        n_sample=n_sample,
        n_pair=n_pair,
        source=source,
        TR_result=val_TR_result,
        sampling=sampling
    )

    ### model train
    model=ModelTrainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_sample_list=val_sample_list,
        TR_result=train_TR_result,
        **model_config
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer","ReaCH-TGN"],
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0005,0.0001],default=0.0005)
    parser.add_argument("--sampling",
        type=str,
        choices=[
            "dependent",
            "independent"
        ],
        default="dependent"
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
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "sampling":args.sampling,
        "dataset_name":args.dataset_name,
        "source":args.source
    }
    main(**app_config)