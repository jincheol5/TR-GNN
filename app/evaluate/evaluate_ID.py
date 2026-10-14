import argparse
from statistics import mean
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,ModelUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer,ReaCH_TGN,TR_GNN
from model_train import ModelTrainer,ReaCH_TGN_Trainer,TR_GNN_Trainer

def main(**kwargs):
    """
    Init setting
    """
    ### set parameter
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
    batch_size=200
    msg_fn="mlp"
    aggr_fn="mean"

    ### set test graph_loader
    dataset_name=kwargs["dataset_name"]
    data=DataUtils.preprocess_snap_dataset(dataset_name=dataset_name)
    graph_df=data["graph_df"]
    graph_dataset=TemporalGraphDataset(df=graph_df)
    graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
    _,_,test_loader=TrainUtils.split_data_loader(data_loader=graph_loader)
    
    """
    Set case
    """
    model_name_list=["TGAT","TGN","DyGFormer","ReaCH-TGN","TR-GNN"]
    seed_list=[1,2,3]
    lr_list=[0.0001,0.0005]
    match dataset_name:
        case "CollegeMsg":
            source_list=[259,319,673,86,274,711,272,554,617,439]
        case "bitcoin-alpha":
            source_list=[323,566,402,717,1402,1216,1212,440,1928,299]
        case "bitcoin-otc":
            source_list=[89,273,212,198,42,57,107,147,344,45]
    sampling_list=["random","focused"]

    """
    Evaluate model
    """
    model_averages={}
    for model_name in model_name_list:
        sampling_scores={
            sampling:{"base_acc":[], "range_1_acc":[], "range_2_acc":[], "range_3_acc":[]}
            for sampling in sampling_list
            if not (model_name=="TR-GNN" and sampling=="random")
        }
        for seed in seed_list:
            for lr in lr_list:
                for source in source_list:
                    for sampling in sampling_list:
                        ### TR-GNN은 focused sampling만 학습
                        if model_name=="TR-GNN" and sampling=="random":
                            continue

                        ### set seed
                        TrainUtils.set_seed(seed=seed)

                        ### set graph, model
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
                                ### TGN 하이퍼 파라미터
                                msg_fn=f"mlp"
                                aggr_fn=f"mean"
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
                                ### DyGFormer 하이퍼 파라미터
                                co_dim=32
                                common_dim=32
                                max_history_len=10
                                patch_size=5
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

                        ### load model
                        file_name=f"{model_name}_{dataset_name}_S{seed}_LR{lr}_B{batch_size}_SRC{source}_{sampling}"
                        model=ModelUtils.load_ID_model_parameter(
                            model=model,
                            file_name=file_name,
                            dataset_name=dataset_name
                        )

                        ### set test sample_loader
                        base_test_sample_loader=DataUtils.load_ID_test_sample_loader(
                            dataset_name=dataset_name,
                            source=source,
                            evaluate_type=f"base"
                        )
                        hop_range_test_sample_loader=DataUtils.load_ID_test_sample_loader(
                            dataset_name=dataset_name,
                            source=source,
                            evaluate_type=f"hop_range"
                        )

                        ### set model_config
                        model_config={
                            "model_name":model_name,
                            "source":source
                        }

                        ### evaluate base
                        match model_name:
                            case "TGAT"|"TGN"|"DyGFormer":
                                evaluate_result=ModelTrainer.evaluate(
                                    model=model,
                                    test_loader=test_loader,
                                    test_sample_loader=base_test_sample_loader,
                                    **model_config
                                )
                            case "ReaCH-TGN":
                                evaluate_result=ReaCH_TGN_Trainer.evaluate(
                                    model=model,
                                    test_loader=test_loader,
                                    test_sample_loader=base_test_sample_loader,
                                    **model_config
                                )
                            case "TR-GNN":
                                evaluate_result=TR_GNN_Trainer.evaluate(
                                    model=model,
                                    test_loader=test_loader,
                                    test_sample_loader=base_test_sample_loader,
                                    **model_config
                                )
                        base_acc=evaluate_result['acc']

                        ### evaluate hop_range
                        match model_name:
                            case "TGAT"|"TGN"|"DyGFormer":
                                evaluate_result=ModelTrainer.evaluate_hop_range(
                                    model=model,
                                    test_loader=test_loader,
                                    test_sample_loader=hop_range_test_sample_loader,
                                    **model_config
                                )
                            case "ReaCH-TGN":
                                evaluate_result=ReaCH_TGN_Trainer.evaluate_hop_range(
                                    model=model,
                                    test_loader=test_loader,
                                    test_sample_loader=hop_range_test_sample_loader,
                                    **model_config
                                )
                            case "TR-GNN":
                                evaluate_result=TR_GNN_Trainer.evaluate_hop_range(
                                    model=model,
                                    test_loader=test_loader,
                                    test_sample_loader=hop_range_test_sample_loader,
                                    **model_config
                                )
                        range_1_acc=evaluate_result['range_1_acc']
                        range_2_acc=evaluate_result['range_2_acc']
                        range_3_acc=evaluate_result['range_3_acc']
                        scores=sampling_scores[sampling]
                        for metric,value in zip(scores,(base_acc,range_1_acc,range_2_acc,range_3_acc)):
                            scores[metric].append(float(value))

        model_averages[model_name]={}
        for sampling,scores in sampling_scores.items():
            result={metric:mean(values) for metric,values in scores.items()}
            result["n_runs"]=len(scores["base_acc"])
            model_averages[model_name][sampling]=result

    print(f"\n{dataset_name}: mean accuracy over source, seed and lr")
    print("\nBase evaluation")
    print(f"{'Model':<14} {'Sampling':<10} {'Runs':>5} {'base_acc':>12}")
    for model_name,sampling_results in model_averages.items():
        for sampling,result in sampling_results.items():
            print(f"{model_name:<14} {sampling:<10} {result['n_runs']:>5} {result['base_acc']:>12.6f}")

    print("\nHop range evaluation")
    print(f"{'Model':<14} {'Sampling':<10} {'Runs':>5} {'range_1_acc':>12} {'range_2_acc':>12} {'range_3_acc':>12}")
    for model_name,sampling_results in model_averages.items():
        for sampling,result in sampling_results.items():
            print(
                f"{model_name:<14} {sampling:<10} {result['n_runs']:>5} "
                f"{result['range_1_acc']:>12.6f} "
                f"{result['range_2_acc']:>12.6f} {result['range_3_acc']:>12.6f}"
            )
    return model_averages

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
