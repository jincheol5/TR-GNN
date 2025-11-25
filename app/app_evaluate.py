import os
import random
import numpy as np
import argparse
import wandb
import torch
from trgnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN,TRGNN,TRGAT

def app_evaluate(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            evaluate models
                test_20
                test_50
                test_100
                test_500
                test_1000
            """
            wandb.init(project="tRGNN",name=f"test_{config['num_nodes']}_result")

            """
            load dataset_list and evaluate
            """
            model_list=['tgat','tgn','trgnn','trgat']
            seed_list=[1,2,3]
            lr_list=[0.001,0.0005]
            batch_size=16
            latent_dim=32

            if config['num_nodes']<1000: # 20,50,100,500
                dataset_list=DataUtils.load_from_pickle(file_name=f"test_{config['num_nodes']}",dir_type="test",num_nodes=config['num_nodes'])
                test_data_loader_list=[]
                for dataset in dataset_list:
                    data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=batch_size)
                    test_data_loader_list.append(data_loader)

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
                                    model_name=f"tgat_{seed}_{lr}_{batch_size}"
                                    trained_model=TGAT(node_dim=1,latent_dim=latent_dim)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                                case 'tgn':
                                    model_name=f"tgn_{emb}_{seed}_{lr}_{batch_size}"
                                    trained_model=TGN(node_dim=1,latent_dim=latent_dim,emb=emb)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                                case 'trgnn':
                                    model_name=f"trgnn_{seed}_{lr}_{batch_size}"
                                    trained_model=TRGNN(node_dim=1,latent_dim=latent_dim)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                                case 'trgat':
                                    model_name=f"trgat_{seed}_{lr}_{batch_size}"
                                    trained_model=TRGAT(node_dim=1,latent_dim=latent_dim)
                                    trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                            
                            if config['num_nodes']<1000: # 20,50,100,500
                                acc,macrof1,auroc,prauc,mcc=ModelTrainer.test(model=trained_model,data_loader_list=test_data_loader_list)
                            else:
                                case_config={
                                    'num_nodes':config['num_nodes'],
                                    'chunk_size':config['chunk_size'],
                                    'batch_size':batch_size
                                }
                                acc,macrof1,auroc,prauc,mcc=ModelTrainer.test_chunk(model=trained_model,config=case_config)

                            wandb.log({
                                f"acc":acc,
                                f"macrof1":macrof1,
                                f"auroc":auroc,
                                f"prauc":prauc,
                                f"mcc":mcc,
                                f"model":model,
                                f"emb": emb if emb else "default",
                                f"seed":seed,
                                f"lr":lr,
                                f"batch_size":batch_size
                            })
        case 2:
            """
            App 2.
            evaluate models about batch_size
                test_20
                test_50
                test_100
                test_500
                test_1000
            """
            wandb.init(project="tRGNN",name=f"test_{config['num_nodes']}_batch_size_result")

            """
            load dataset_list and evaluate
            """
            model_list=['tgn','trgnn','trgat']
            batch_size_list=[4,8,16,32,64]
            seed=1
            lr=0.0005
            latent_dim=32

            if config['num_nodes']<1000: # 20,50,100,500
                dataset_list=DataUtils.load_from_pickle(file_name=f"test_{config['num_nodes']}",dir_type="test",num_nodes=config['num_nodes'])
                test_data_loader_list=[]
                for dataset in dataset_list:
                    data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=batch_size)
                    test_data_loader_list.append(data_loader)

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

            for batch_size in batch_size_list:
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
                            case 'tgn':
                                model_name=f"tgn_{emb}_{seed}_{lr}_{batch_size}"
                                trained_model=TGN(node_dim=1,latent_dim=latent_dim,emb=emb)
                                trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                            case 'trgnn':
                                model_name=f"trgnn_{seed}_{lr}_{batch_size}"
                                trained_model=TRGNN(node_dim=1,latent_dim=latent_dim)
                                trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)
                            case 'trgat':
                                model_name=f"trgat_{seed}_{lr}_{batch_size}"
                                trained_model=TRGAT(node_dim=1,latent_dim=latent_dim)
                                trained_model=DataUtils.load_model_parameter(model=trained_model,model_name=model_name)

                        if config['num_nodes']<1000: # 20,50,100,500
                            acc,macrof1,auroc,prauc,mcc=ModelTrainer.test(model=trained_model,data_loader_list=test_data_loader_list)
                        else:
                            case_config={
                                'num_nodes':config['num_nodes'],
                                'chunk_size':config['chunk_size'],
                                'batch_size':batch_size
                            }
                            acc,macrof1,auroc,prauc,mcc=ModelTrainer.test_chunk(model=trained_model,config=case_config)

                        wandb.log({
                            f"acc":acc,
                            f"macrof1":macrof1,
                            f"auroc":auroc,
                            f"prauc":prauc,
                            f"mcc":mcc,
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
    parser.add_argument("--chunk_size",type=int,default=10)
    args=parser.parse_args()

    config={
        'app_num':args.app_num,
        'num_nodes':args.num_nodes,
        'chunk_size':args.chunk_size
    }
    app_evaluate(config=config)