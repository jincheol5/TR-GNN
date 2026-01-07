import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tr_gnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN

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
            ### wandb
            if config['wandb']:
                if config['model']=='tgn':
                    wandb.init(project="TR_GNN",name=f"{config['model']}_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                else: # tgat
                    wandb.init(project="TR_GNN",name=f"{config['model']}_{config['seed']}_{config['lr']}_{config['batch_size']}")

            ### data load
            train_20_datastream_list=DataUtils.load_from_pickle(file_name=f"train_20_datastream_list",dir_type=f"dataset",mode=f"train",num_nodes=20)
            train_20_trajs_list=DataUtils.load_from_pickle(file_name=f"train_20_trajs_list",dir_type=f"dataset",mode=f"train",num_nodes=20)
            train_data_loader_list=[]
            node_list=[i for i in range(20)]
            for datastream,trajs in zip(train_20_datastream_list,train_20_trajs_list):
                selected_src=random.choice(node_list) # 랜덤하게 source node 선택
                traj=trajs[selected_src]
                train_data_loader=ModelTrainUtils.get_data_loader(datastream=datastream,traj=traj,source_id=selected_src,batch_size=config['batch_size'])
                train_data_loader_list.append(train_data_loader)

            val_20_datastream_list=DataUtils.load_from_pickle(file_name=f"val_20_datastream_list",dir_type=f"dataset",mode=f"val",num_nodes=20)
            val_20_trajs_list=DataUtils.load_from_pickle(file_name=f"val_20_trajs_list",dir_type=f"dataset",mode=f"val",num_nodes=20)
            val_data_loader_list=[]
            for datastream,trajs in zip(val_20_datastream_list,val_20_trajs_list):
                for src_id,traj in enumerate(trajs):
                    val_data_loader=ModelTrainUtils.get_data_loader(datastream=datastream,traj=traj,source_id=src_id,batch_size=config['batch_size'])
                    val_data_loader_list.append(val_data_loader)

            ### model train
            is_memory=False
            match config['model']:
                case 'tgat':
                    model=TGAT(traj_dim=1,latent_dim=config['latent_dim'])
                    is_memory=False
                case 'tgn':
                    model=TGN(traj_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
                    is_memory=True
            ModelTrainer.train(model=model,is_memory=is_memory,train_data_loader_list=train_data_loader_list,val_data_loader_list=val_data_loader_list,config=config)
            
            if config['wandb']:
                wandb.finish()
            
            ### save model
            if config['save_model']:
                match config['model']:
                    case 'tgat':
                        model_name=f"{config['model']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                        DataUtils.save_model_parameter(model=model,model_name=model_name)
                    case 'tgn':
                        model_name=f"{config['model']}_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                        DataUtils.save_model_parameter(model=model,model_name=model_name)

        case 2:
            """
            App 2.
            test model
            """
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
                        test_data_loader=ModelTrainUtils.get_data_loader(datastream=datastream,traj=traj,source_id=src,batch_size=config['batch_size'])
                        test_data_loader_list.append(test_data_loader)
            
            ### model test
            is_memory=False
            match config['model']:
                case 'tgat':
                    model_name=f"{config['model']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TGAT(traj_dim=1,latent_dim=config['latent_dim'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                case 'tgn':
                    model_name=f"{config['model']}_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    model=TGN(traj_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                    is_memory=True
            perform=ModelTrainer.test(model=model,is_memory=is_memory,data_loader_list=test_data_loader_list)
            print(f"Evaluate {model_name} TR Acc: {perform['acc']} Macro-f1: {perform['macrof1']} PR-AUC: {perform['prauc']} MCC: {perform['mcc']}")


if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    
    # setting
    parser.add_argument("--model",type=str,default='tgat') # tgat,tgn
    parser.add_argument("--emb",type=str,default='attn') # time, sum, attn

    # train
    parser.add_argument("--optimizer",type=str,default='adam') # adam, sgd
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--early_stop",type=int,default=1)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--seed",type=int,default=1) # 1, 2, 3
    parser.add_argument("--lr",type=float,default=0.0005) # 0.0005, 0,0001
    parser.add_argument("--batch_size",type=int,default=16) # 4,8,16,32,64
    parser.add_argument("--latent_dim",type=int,default=32)
    
    # 학습 로그 및 저장
    parser.add_argument("--wandb",type=int,default=0)
    parser.add_argument("--save_model",type=int,default=0)

    # 평가
    parser.add_argument("--num_nodes",type=int,default=20)
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
        'num_nodes':args.num_nodes
    }
    app_train(config=config)