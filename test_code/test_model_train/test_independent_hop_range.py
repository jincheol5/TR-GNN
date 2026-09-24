import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer
from model_train import ModelTrainer

"""
<< Test >> 
model_train.ModelTrainer

Test Model:
    - TGAT
    - TGN
    - DyGFormer
    - ReaCH-TGN
"""
def test_fn(**kwargs):
    ### set dataset and graph
    data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
    graph_df=data["graph_df"]
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    node_dim=data["node_dim"]
    edge_dim=data["edge_dim"]
    graph=TGN_Graph(
        graph_df=graph_df,
        node_ft=node_ft,
        edge_ft=edge_ft,
        node_dim=node_dim,
        edge_dim=edge_dim
    )

    ### set random seed
    seed=1
    TrainUtils.set_seed(seed=seed)
    graph.set_random_seed(seed=seed)

    ### TR sample 관련 파라미터
    batch_size=200
    n_pair=10
    sampling=f"independent"

    ### 모델 관련 파라미터
    n_layer=1
    n_neighbor=10
    n_head=4

    ### 학습 관련 파라미터
    time_dim=32
    latent_dim=32
    embed_dim=32
    epoch=100
    lr=0.0005
    optimizer=f"adam"
    early_stop=True
    patience=10

    ### set data_loader
    train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
    train_dataset=TemporalGraphDataset(df=train_df)
    val_dataset=TemporalGraphDataset(df=val_df)
    test_dataset=TemporalGraphDataset(df=test_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    ### load SR, TR result
    train_TR_result=DataUtils.load_TR_result(
        dataset_name=f"enron",
        batch_size=batch_size,
        purpose="train"
    )
    val_TR_result=DataUtils.load_TR_result(
        dataset_name=f"enron",
        batch_size=batch_size,
        purpose="val"
    )
    test_TR_result=DataUtils.load_TR_result(
        dataset_name=f"enron",
        batch_size=batch_size,
        purpose="test"
    )

    ### set val_sample_list using TR sampling
    val_sample_list=TrainUtils.get_TR_sample_list(
        n_pair=n_pair,
        data_loader=val_loader,
        TR_result=val_TR_result,
        sampling=f"independent"
    )

    ### set test_sample_list
    test_TR_label=test_TR_result["label"]
    source_candidates=TrainUtils.get_source_candidates(n_source=1,TR_label=test_TR_label)
    source=source_candidates[0]
    test_sample_list=TrainUtils.get_TR_sample_list(
        data_loader=test_loader,
        n_pair=100,
        source=source,
        TR_result=test_TR_result,
        sampling=f"hop_range"
    )

    ### model별 수행
    match kwargs["model_name"]:
        case "TGAT":
            """
            Test Model: TGAT
            """
            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
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
            model=ModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=ModelTrainer.evaluate_hop_range(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']}: {evaluate_result['acc']}")
            print(f"Evaluate hop_range_1 (1<=hop<5) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_1_acc']}")
            print(f"Evaluate hop_range_2 (5<=hop<10) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_2_acc']}")
            print(f"Evaluate hop_range_3 (10<=hop) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_3_acc']}")

        case "TGN":
            ### TGN 학습 관련 파라미터
            msg_dim=32
            mem_dim=32

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "msg_dim":msg_dim,
                "mem_dim":mem_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
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
                n_head=n_head
            )   
            model=ModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=ModelTrainer.evaluate_hop_range(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']}: {evaluate_result['acc']}")
            print(f"Evaluate hop_range_1 (1<=hop<5) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_1_acc']}")
            print(f"Evaluate hop_range_2 (5<=hop<10) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_2_acc']}")
            print(f"Evaluate hop_range_3 (10<=hop) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_3_acc']}")

        case "DyGFormer":
            ### set DyGFormer graph
            graph=DyGFormer_Graph(
                graph_df=graph_df,
                node_ft=node_ft,
                edge_ft=edge_ft,
                node_dim=node_dim,
                edge_dim=edge_dim
            )
            seed=1
            graph.set_random_seed(seed=seed)

            ### DyGFormer 모델 관련 파라미터
            max_seq_len=10
            patch_size=5

            ### DyGFormer 학습 관련 파라미터
            co_dim=32
            common_dim=32

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "max_seq_len":max_seq_len,
                "patch_size":patch_size,
                "latent_dim":latent_dim,
                "time_dim":time_dim,
                "co_dim":co_dim,
                "common_dim":common_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                patch_size=patch_size,
                graph=graph,
                n_neighbor=n_neighbor,
                n_layer=n_layer,
                n_head=n_head
            )
            model=ModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=ModelTrainer.evaluate_hop_range(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']}: {evaluate_result['acc']}")
            print(f"Evaluate hop_range_1 (1<=hop<5) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_1_acc']}")
            print(f"Evaluate hop_range_2 (5<=hop<10) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_2_acc']}")
            print(f"Evaluate hop_range_3 (10<=hop) ACC of {kwargs['model_name']}: {evaluate_result['hop_range_3_acc']}")

        case "ReaCH-TGN":
            ### ReaCH-TGN 학습 관련 파라미터
            msg_dim=32
            mem_dim=32

            # ### model config
            # model_config={
            #     "model_name":kwargs["model_name"],
            #     "seed":seed,
            #     "batch_size":batch_size,
            #     "max_hop":max_hop,
            #     "n_pair":n_pair,
            #     "sampling":sampling,
            #     "pos_hard_ratio":pos_hard_ratio,
            #     "neg_hard_ratio":neg_hard_ratio,
            #     "n_layer":n_layer,
            #     "n_neighbor":n_neighbor,
            #     "n_head":n_head,
            #     "time_dim":time_dim,
            #     "latent_dim":latent_dim,
            #     "msg_dim":msg_dim,
            #     "mem_dim":mem_dim,
            #     "embed_dim":embed_dim,
            #     "epoch":epoch,
            #     "lr":lr,
            #     "optimizer":optimizer,
            #     "early_stop":early_stop,
            #     "patience":patience
            # }

            # ### set model and train
            # model=ReaCH_TGN(
            #     node_dim=node_dim,
            #     edge_dim=edge_dim,
            #     time_dim=time_dim,
            #     msg_dim=msg_dim,
            #     mem_dim=mem_dim,
            #     latent_dim=latent_dim,
            #     embed_dim=embed_dim,
            #     graph=graph,
            #     n_layer=n_layer,
            #     n_neighbor=n_neighbor,
            #     n_head=n_head
            # )
            # model=ReaCH_TGN_Trainer.train(
            #     model=model,
            #     train_loader=train_loader,
            #     val_loader=val_loader,
            #     val_sample_list=val_sample_list,
            #     # SR_result=train_SR_result,
            #     TR_result=train_TR_result,
            #     **model_config
            # )
            # evaluate_result=ReaCH_TGN_Trainer.evaluate(
            #     model=model,
            #     val_loader=val_loader,
            #     test_loader=test_loader,
            #     test_sample_list=test_sample_list,
            #     **model_config
            # )
            # print(f"Evaluate ACC of {kwargs['model_name']} using {kwargs['sampling']} TR Sampling: {evaluate_result['acc']}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer","ReaCH-TGN"],
        default=f"TGAT"
    )
    args=parser.parse_args()
    test_config={
        "model_name":args.model_name
    }
    test_fn(**test_config)