import os
import random
import pickle
import networkx as nx
import torch
from tqdm import tqdm
from typing_extensions import Literal
import requests
import gzip
import io

class DataUtils:
    dataset_path=os.path.join('..','data','tr-gnn')
    @staticmethod
    def save_to_pickle(data,file_name:str,dir_type:Literal['graph','dataset'],dataset_name:Literal['CollegeMsg','bitcoin_otc','bitcoin_alpha'],mode:Literal['train','val','test']='train'):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='dataset':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,dataset_name,mode,file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")