import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tr_gnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN,TR_GNN,TR_GAT

val_20_datastream_list=DataUtils.load_from_pickle(file_name=f"val_20_datastream_list",dir_type=f"dataset",mode=f"val",num_nodes=20)
val_20_trajs_list=DataUtils.load_from_pickle(file_name=f"val_20_trajs_list",dir_type=f"dataset",mode=f"val",num_nodes=20)

datastream=val_20_datastream_list[0]
traj=val_20_trajs_list[0][0]

data_loader=ModelTrainUtils.get_data_loader(datastream=datastream,traj=traj,source_id=0,batch_size=1)

batch=data_loader[0]

print(f"init_traj: {batch['init_traj']}")