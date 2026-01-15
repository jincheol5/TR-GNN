import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tr_gnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN,TR_GNN

def app_evaluate(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            Evaluate: size-OOD
            """
            ### set wandb
            wandb.init(project="TR_GNN",name=f"test_{config['num_nodes']}_result")

            ### set parameters
            model_list=['tgat','tgn','trgnn']
            seed_list=[1,2,3]
            lr_list=[0.0001,0.0005]
            batch_size=16
            latent_dim=32

            ### data load
            test_data_loader_list=[]
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            for graph_type in tqdm(graph_type_list,desc=f"Load datasets..."):
                for graph_id in range(5):
                    datastream=DataUtils.load_from_pickle(
                        file_name=f"test_{config['num_nodes']}_{graph_type}_{graph_id}_datastream",
                        dir_type=f"dataset",
                        mode="test",
                        num_nodes=config['num_nodes'],
                        is_print=False
                    )
                    src_list=DataUtils.load_from_pickle(
                        file_name=f"test_{config['num_nodes']}_{graph_type}_{graph_id}_src_list",
                        dir_type=f"dataset",
                        mode="test",
                        num_nodes=config['num_nodes'],
                        is_print=False
                    )
                    for src in src_list:
                        traj=DataUtils.load_from_pickle(
                            file_name=f"test_{config['num_nodes']}_{graph_type}_{graph_id}_traj_{src}",
                            dir_type=f"dataset",
                            mode="test",
                            num_nodes=config['num_nodes'],
                            is_print=False
                        )
                        test_data_loader=ModelTrainUtils.get_data_loader(datastream=datastream,traj=traj,source_id=src,batch_size=batch_size)
                        test_data_loader_list.append(test_data_loader)

            ### evaluate
            for seed in seed_list:
                """
                seed setting
                """
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed) 
                os.environ["PYTHONHASHSEED"]=str(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic=True 
                torch.backends.cudnn.benchmark=False

                for lr in lr_list:
                    for model in model_list:
                        if model=='tgn':
                            emb_list=['time','sum','attn']
                        else:
                            emb_list=[None]
                        for emb in emb_list:
                            """
                            model setting and evaluating
                            """
                            match model:
                                case 'tgat':
                                    model_name=f"{model}_{seed}_{lr}_{batch_size}"
                                    trained_model=TGAT(latent_dim=latent_dim)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                                case 'tgn':
                                    model_name=f"{model}_{emb}_{seed}_{lr}_{batch_size}"
                                    trained_model=TGN(latent_dim=latent_dim,emb=emb)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                                case 'trgnn':
                                    model_name=f"{model}_{seed}_{lr}_{batch_size}"
                                    trained_model=TR_GNN(latent_dim=latent_dim)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)

                            perform=ModelTrainer.test(model=trained_model,data_loader_list=test_data_loader_list)

                            wandb.log({
                                f"acc":perform['acc'],
                                f"macrof1":perform['macrof1'],
                                f"prauc":perform['prauc'],
                                f"mcc":perform['mcc'],
                                f"model":model,
                                f"emb": emb if emb else "default",
                                f"seed":seed,
                                f"lr":lr,
                                f"batch_size":batch_size
                            })
        case 2:
            """
            App 2.
            Evaluate: batch-size sensitivity
            """
            ### set wandb
            wandb.init(project="TR_GNN",name=f"test_{config['num_nodes']}_batch_result")

            ### set parameters
            model_list=['tgn','trgnn']
            emb='attn'
            seed=1
            lr=0.0005
            batch_size_list=[4,8,16,32,64]
            latent_dim=32

            ### data load
            test_data_loader_list=[]
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            for graph_type in tqdm(graph_type_list,desc=f"Load datasets..."):
                for graph_id in range(5):
                    datastream=DataUtils.load_from_pickle(
                        file_name=f"test_{config['num_nodes']}_{graph_type}_{graph_id}_datastream",
                        dir_type=f"dataset",
                        mode="test",
                        num_nodes=config['num_nodes'],
                        is_print=False
                    )
                    src_list=DataUtils.load_from_pickle(
                        file_name=f"test_{config['num_nodes']}_{graph_type}_{graph_id}_src_list",
                        dir_type=f"dataset",
                        mode="test",
                        num_nodes=config['num_nodes'],
                        is_print=False
                    )
                    for src in src_list:
                        traj=DataUtils.load_from_pickle(
                            file_name=f"test_{config['num_nodes']}_{graph_type}_{graph_id}_traj_{src}",
                            dir_type=f"dataset",
                            mode="test",
                            num_nodes=config['num_nodes'],
                            is_print=False
                        )
                        test_data_loader=ModelTrainUtils.get_data_loader(datastream=datastream,traj=traj,source_id=src,batch_size=batch_size)
                        test_data_loader_list.append(test_data_loader)

            ### evaluate
            """
            seed setting
            """
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) 
            os.environ["PYTHONHASHSEED"]=str(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic=True 
            torch.backends.cudnn.benchmark=False
            
            for model in model_list:
                for batch_size in batch_size_list:
                    """
                    model setting and evaluating
                    """
                    match model:
                        case 'tgn':
                            model_name=f"{model}_attn_{seed}_{lr}_{batch_size}"
                            trained_model=TGN(latent_dim=latent_dim,emb=emb)
                            trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                        case 'trgnn':
                            model_name=f"{model}_{seed}_{lr}_{batch_size}"
                            trained_model=TR_GNN(latent_dim=latent_dim)
                            trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)

                    perform=ModelTrainer.test(model=trained_model,data_loader_list=test_data_loader_list)

                    wandb.log({
                        f"acc":perform['acc'],
                        f"macrof1":perform['macrof1'],
                        f"prauc":perform['prauc'],
                        f"mcc":perform['mcc'],
                        f"model":model,
                        f"emb": emb if emb else "default",
                        f"seed":seed,
                        f"lr":lr,
                        f"batch_size":batch_size
                    })

if __name__=="__main__":
    """
    Execute app_evaluate
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        'app_num':args.app_num,
        'num_nodes':args.num_nodes
    }
    app_evaluate(config=config)