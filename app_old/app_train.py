import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from trgnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN,TRGNN,TRGAT

def app_train(config: dict):
    """
    seed setting
    """
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    os.environ["PYTHONHASHSEED"]=str(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic=True 
    torch.backends.cudnn.benchmark=False

    match config['app_num']:
        case 1:
            """
            App 1.
            train model
            """
            """
            wandb
            """
            if config['wandb']:
                if config['model']=='tgn':
                    wandb.init(project="TR-GNN",name=f"{config['model']}_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                else: # tgat, trgnn, trgat
                    wandb.init(project="TR-GNN",name=f"{config['model']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                wandb.config.update(config)

            """
            data loading
            """
            dataset_list=DataUtils.load_from_pickle(file_name=f"train_20",dir_type="train")
            selected_dataset_list=[]
            for i in range(0,len(dataset_list),20):
                sub_dataset_list=dataset_list[i:i+20] 
                if len(sub_dataset_list):
                    selected_dataset_list.append(random.choice(sub_dataset_list))

            train_data_loader_list=[]
            for dataset in selected_dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                train_data_loader_list.append(data_loader)
            
            dataset_list=DataUtils.load_from_pickle(file_name=f"val_20",dir_type="val")
            val_data_loader_list=[]
            for dataset in dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                val_data_loader_list.append(data_loader)

            """
            model setting and training
            """
            match config['model']:
                case 'tgat':
                    model=TGAT(node_dim=1,latent_dim=config['latent_dim'])
                case 'tgn':
                    model=TGN(node_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
                case 'trgnn':
                    model=TRGNN(node_dim=1,latent_dim=config['latent_dim'])
                case 'trgat':
                    model=TRGAT(node_dim=1,latent_dim=config['latent_dim'])
            ModelTrainer.train(model=model,train_data_loader_list=train_data_loader_list,val_data_loader_list=val_data_loader_list,validate=True,config=config)

            """
            save model
            """
            if config['save_model']:
                match config['model']:
                    case 'tgat':
                        model_name=f"tgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    case 'tgn':
                        model_name=f"tgn_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    case 'trgnn':
                        model_name=f"trgnn_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    case 'trgat':
                        model_name=f"trgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                DataUtils.save_model_parameter(model=model,model_name=model_name)

        case 2:
            """
            App 2. 
            test model
            """
            """
            data loading
            """
            dataset_list=DataUtils.load_from_pickle(file_name=f"test_{config['num_nodes']}",dir_type="test",num_nodes=config['num_nodes'])
            test_data_loader_list=[]
            for dataset in dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                test_data_loader_list.append(data_loader)

            """
            model setting and evaluating
            """
            match config['model']:
                case 'tgat':
                    model_name=f"tgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TGAT(node_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'tgn':
                    model_name=f"tgn_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TGN(node_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'trgnn':
                    model_name=f"trgnn_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TRGNN(node_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'trgat':
                    model_name=f"trgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TRGAT(node_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
            acc,macrof1,auroc,prauc,mcc=ModelTrainer.test(model=model,data_loader_list=test_data_loader_list)
            print(f"test_{config['num_nodes']} tR Acc: {acc} macro-f1: {macrof1} AUROC: {auroc} PR-AUC: {prauc} MCC: {mcc}")

        case 3:
            """
            App 3.
            test model
            chunk files
            """
            """
            model setting and evaluating
            """
            match config['model']:
                case 'tgat':
                    model_name=f"tgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TGAT(node_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'tgn':
                    model_name=f"tgn_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TGN(node_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'trgnn':
                    model_name=f"trgnn_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TRGNN(node_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'trgat':
                    model_name=f"trgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TRGAT(node_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
            acc,macrof1,auroc,prauc,mcc=ModelTrainer.test_chunk(model=model,config=config)
            print(f"test_{config['num_nodes']} tR Acc: {acc} macro-f1: {macrof1} AUROC: {auroc} PR-AUC: {prauc} MCC: {mcc}")

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    
    # setting
    parser.add_argument("--model",type=str,default='tgat') # tgat,tgn,trgnn,trgat
    parser.add_argument("--emb",type=str,default='attn') # time, attn, sum

    # train
    parser.add_argument("--optimizer",type=str,default='adam') # adam, sgd
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--early_stop",type=int,default=1)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--seed",type=int,default=1) # 1, 2, 3
    parser.add_argument("--lr",type=float,default=0.0005) # 0.001, 0,0005
    parser.add_argument("--batch_size",type=int,default=32) # 32, 64
    parser.add_argument("--latent_dim",type=int,default=32)
    
    # 학습 로그 및 저장
    parser.add_argument("--wandb",type=int,default=0)
    parser.add_argument("--save_model",type=int,default=0)

    # 평가
    parser.add_argument("--num_nodes",type=int,default=20)
    parser.add_argument("--chunk_size",type=int,default=10)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        # setting
        'model':args.model,
        'emb':args.emb,
        # train
        'optimizer':args.optimizer,
        'epochs':args.epochs,
        'early_stop':args.early_stop,
        'patience':args.patience,
        'seed':args.seed,
        'lr':args.lr,
        'batch_size':args.batch_size,
        'latent_dim':args.latent_dim,
        # 학습 로그 및 저장
        'wandb':args.wandb,
        'save_model':args.save_model,
        # 평가
        'num_nodes':args.num_nodes,
        'chunk_size':args.chunk_size
    }
    app_train(config=config)