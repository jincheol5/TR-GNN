import argparse
from utils import DataUtils,TrainUtils,ModelUtils
from model import TGAT,TGN,DyGFormer,ReaCH_TGN,TR_GNN
from model_train import ModelTrainer_OOD,ReaCH_TGN_Trainer_OOD,TR_GNN_Trainer_OOD

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
    msg_fn=f"mlp"
    aggr_fn=f"mean"

    ### 학습 관련 파라미터
    optimizer=f"adam"
    epoch=100
    early_stop=True
    patience=10
    batch_size=32
    lr=kwargs["lr"]

    ### set train_graph_list, train_loader_list, train_TR_result_list
    graph_type_list=[
        "ladder",
        "grid",
        "tree",
        "erdos_renyi",
        "barabasi_albert",
        "community",
        "caveman"
    ]
    train_graph_list=[]
    train_loader_list=[]
    train_TR_result_list=[]
    for graph_type in graph_type_list:
        # get train_graph_list, train_loader_list
        file_name=f"{graph_type}_train_N{n_node}"
        graph_df_list=DataUtils.load_graph_df_list_pickle(
            file_name=file_name,
            purpose=f"train"
        )
        graph_list=TrainUtils.get_graph_list(
            graph_df_list=graph_df_list,
            model_name=model_name,
            node_dim=node_dim,
            edge_dim=edge_dim
        )
        train_graph_list.extend(graph_list)
        graph_loader_list=TrainUtils.get_graph_loader_list(graph_df_list=graph_df_list,batch_size=batch_size)   
        train_loader_list.extend(graph_loader_list)

        # get train_TR_result_list
        file_name=f"{graph_type}_train_N{n_node}_B{batch_size}"
        TR_result_list=DataUtils.load_TR_result_list(
            file_name=file_name,
            purpose=f"train"
        )
        train_TR_result_list.extend(TR_result_list)

    ### set val_graph_list, val_loader_list
    val_graph_list=[]
    val_loader_list=[]
    val_TR_result_list=[]
    for graph_type in graph_type_list:
        # get val_graph_list, val_loader_list
        file_name=f"{graph_type}_val_N{n_node}"
        graph_df_list=DataUtils.load_graph_df_list_pickle(
            file_name=file_name,
            purpose=f"val"
        )
        graph_list=TrainUtils.get_graph_list(
            graph_df_list=graph_df_list,
            model_name=model_name,
            node_dim=node_dim,
            edge_dim=edge_dim
        )
        val_graph_list.extend(graph_list)
        graph_loader_list=TrainUtils.get_graph_loader_list(graph_df_list=graph_df_list,batch_size=batch_size)   
        val_loader_list.extend(graph_loader_list)

        # get val_TR_result_list
        file_name=f"{graph_type}_val_N{n_node}_B{batch_size}"
        TR_result_list=DataUtils.load_TR_result_list(
            file_name=file_name,
            purpose=f"val"
        )
        val_TR_result_list.extend(TR_result_list)

    ### set val_sample_loader_list
    val_sample_loader_list=TrainUtils.get_sample_loader_list(
        n_node=n_node,
        graph_loader_list=val_loader_list,
        TR_result_list=val_TR_result_list
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
                n_head=n_head,
                use_source_info=True
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

    ### set model_config
    model_config={
        "model_name":kwargs["model_name"],
        "n_node":n_node,
        "optimizer":optimizer,
        "epoch":epoch,
        "early_stop":early_stop,
        "patience":patience,
        "batch_size":batch_size,
        "seed":seed,
        "lr":lr
    }

    ### train model
    match model_name:
        case "TGAT"|"TGN"|"DyGFormer":
            model=ModelTrainer_OOD.train(
                model=model,
                train_graph_list=train_graph_list,
                train_loader_list=train_loader_list,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_sample_loader_list=val_sample_loader_list,
                TR_result_list=train_TR_result_list,
                **model_config
            )
        case "ReaCH-TGN":
            model=ReaCH_TGN_Trainer_OOD.train(
                model=model,
                train_graph_list=train_graph_list,
                train_loader_list=train_loader_list,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_sample_loader_list=val_sample_loader_list,
                TR_result_list=train_TR_result_list,
                **model_config
            )
        case "TR-GNN":
            model=TR_GNN_Trainer_OOD.train(
                model=model,
                train_graph_list=train_graph_list,
                train_loader_list=train_loader_list,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_sample_loader_list=val_sample_loader_list,
                TR_result_list=train_TR_result_list,
                **model_config
            )

    ### save model
    if kwargs["save_model"]:
        file_name=f"{model_name}_S{seed}_LR{lr}_B{batch_size}"
        ModelUtils.save_OOD_model_parameter(
            model=model,
            file_name=file_name
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
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "save_model":args.save_model
    }
    main(**app_config)