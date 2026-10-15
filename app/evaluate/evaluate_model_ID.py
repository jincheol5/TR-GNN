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
    sampling=kwargs["sampling"]

    ### TGN 하이퍼 파라미터
    msg_fn=f"mlp"
    aggr_fn=f"mean"

    ### DyGFormer 하이퍼 파라미터
    co_dim=32
    common_dim=32
    max_history_len=10
    patch_size=5

    ### 평가 관련 파라미터
    batch_size=200
    lr=kwargs["lr"]
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

        # 평가 관련 파라미터
        "source":source
    }

    ### set dataset
    dataset_name=kwargs["dataset_name"]
    data=DataUtils.preprocess_snap_dataset(dataset_name=dataset_name)
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

    ### set model
    match model_name:
        case "TGAT":
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
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
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
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
                n_layer=n_layer,
                n_head=n_head,
                max_history_len=max_history_len,
                patch_size=patch_size
            )
        case "ReaCH-TGN":
            model=ReaCH_TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
        case "TR-GNN":
            model=TR_GNN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
    model.set_graph(graph=graph)

    ### Load model
    if sampling=="focused":
        file_name=f"{model_name}_{dataset_name}_S{seed}_LR{lr}_B{batch_size}_SRC{source}_focused"
    else: # random
        file_name=f"{model_name}_{dataset_name}_S{seed}_LR{lr}_B{batch_size}_SRC{source}_random"
    model=ModelUtils.load_ID_model_parameter(
        model=model,
        file_name=file_name,
        dataset_name=dataset_name
    )

    ### set data_loader
    graph_dataset=TemporalGraphDataset(df=graph_df)
    graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
    _,_,test_loader=TrainUtils.split_data_loader(data_loader=graph_loader)

    ### Evaluation
    test_sample_loader=DataUtils.load_ID_test_sample_loader(
        dataset_name=dataset_name,
        source=source
    )
    match model_name:
        case "TGAT"|"TGN"|"DyGFormer":
            evaluate_result=ModelTrainer.evaluate_hop_range(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader,
                **model_config
            )
        case "ReaCH-TGN":
            evaluate_result=ReaCH_TGN_Trainer.evaluate_hop_range(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader,
                **model_config
            )
        case "TR-GNN":
            evaluate_result=TR_GNN_Trainer.evaluate_hop_range(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader,
                **model_config
            )
    print(f"Evaluation Result:")
    print(f"dataset: {dataset_name}")
    print(f"model: {file_name}")
    print(f"ACC: {evaluate_result['acc']}")
    print(f"Hop Range 1 (1 <= Hop < 3) ACC: {evaluate_result['range_1_acc']}")
    print(f"Hop Range 2 (3 <= Hop < 5) ACC: {evaluate_result['range_2_acc']}")
    print(f"Hop Range 3 (5 <= Hop) ACC: {evaluate_result['range_3_acc']}")

    return evaluate_result

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
    parser.add_argument("--sampling",
        type=str,
        choices=[
            "random",
            "focused"
        ],
        default="focused"
    )
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "dataset_name":args.dataset_name,
        "source":args.source,
        "sampling":args.sampling
    }
    main(**app_config)
