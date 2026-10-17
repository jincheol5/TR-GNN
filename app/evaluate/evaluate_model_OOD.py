import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,ModelUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer,ReaCH_TGN,TR_GNN
from model_train import ModelTrainer_OOD,ReaCH_TGN_Trainer_OOD,TR_GNN_Trainer_OOD

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
    msg_fn=f"mlp"
    aggr_fn=f"mean"

    ### 평가 관련 파라미터
    lr=kwargs["lr"]
    n_node=kwargs["n_node"]
    if n_node==100:
        batch_size=32
    else: # 500, 1000
        batch_size=200

    ### set test_graph_list, test_loader_list
    test_data=TrainUtils.get_data_for_evaluate_OOD(
        model_name=model_name,
        node_dim=node_dim,
        edge_dim=edge_dim,
        n_node=n_node,
        batch_size=batch_size,
        purpose=f"test"
    )
    test_graph_list=test_data["graph_list"]
    test_loader_list=test_data["graph_loader_list"]

    ### set test_sample_loader_list
    file_name=f"test_sample_loader_list_N{n_node}_B{batch_size}"
    test_sample_loader_list=DataUtils.load_evaluate_sample_loader_list(
        file_name=file_name,
        purpose=f"test"
    )

    ### set model
    match model_name:
        case "TGAT":
            model=TGAT(
                use_source_info=True,
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
                use_source_info=True,
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
            ### DyGFormer 하이퍼 파라미터
            co_dim=32
            common_dim=32
            max_history_len=10
            patch_size=5
            model=DyGFormer(
                use_source_info=True,
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
                use_source_info=True,
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

    ### Load model
    file_name=f"{model_name}_S{seed}_LR{lr}_B32"
    model=ModelUtils.load_OOD_model_parameter(
        model=model,
        file_name=file_name
    )

    ### set model_config
    model_config={
        "model_name":kwargs["model_name"],
        "n_node":n_node
    }

    ### Evaluation
    match model_name:
        case "TGAT"|"TGN"|"DyGFormer":
            evaluate_result=ModelTrainer_OOD.evaluate(
                model=model,
                test_graph_list=test_graph_list,
                test_loader_list=test_loader_list,
                test_sample_loader_list=test_sample_loader_list,
                **model_config
            )
        case "ReaCH-TGN":
            evaluate_result=ReaCH_TGN_Trainer_OOD.evaluate(
                model=model,
                test_graph_list=test_graph_list,
                test_loader_list=test_loader_list,
                test_sample_loader_list=test_sample_loader_list,
                **model_config
            )
        case "TR-GNN":
            evaluate_result=TR_GNN_Trainer_OOD.evaluate(
                model=model,
                test_graph_list=test_graph_list,
                test_loader_list=test_loader_list,
                test_sample_loader_list=test_sample_loader_list,
                **model_config
            )
    print(f"Evaluation Result:")
    print(f"model: {file_name}")
    print(f"ACC: {evaluate_result['acc']}")
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
    parser.add_argument("--n_node",type=int,choices=[100,500,1000],default=100)
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0005,0.0001],default=0.0005)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "n_node":args.n_node,
        "seed":args.seed,
        "lr":args.lr
    }
    main(**app_config)
