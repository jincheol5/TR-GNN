import os
import torch
import torch.nn as nn
from typing import Literal

BASE_PATH=os.path.join("..","data","TR-GNN","inference")

class ModelUtils:
    @staticmethod
    def save_ID_model_parameter(
            model:nn.Module,
            model_name:str,
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-alpha",
                "bitcoin-otc"
            ]
        ):
        file_name=model_name+".pt"
        file_path=os.path.join(BASE_PATH,f"ID",dataset_name,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(model.state_dict(),file_path)
        print(f"Save {model_name} model parameter!")

    @staticmethod
    def load_ID_model_parameter(
            model:nn.Module,
            model_name:str,
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-alpha",
                "bitcoin-otc"
            ]
        ):
        file_name=model_name+".pt"
        file_path=os.path.join(BASE_PATH,f"ID",dataset_name,file_name)
        model.load_state_dict(torch.load(file_path))
        return model
