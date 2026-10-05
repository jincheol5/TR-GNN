import torch
import torch.nn as nn
from typing import Literal

class Last_State(nn.Module):
    def __init__(self,
            n_node:int
        ):
        """
        Model의 Intermediate Result 저장
        """
        super().__init__()
        self.n_node=n_node
        self.register_buffer(
            "last_state",
            torch.zeros(n_node+1),
        )
        self.init_last_state()

    def init_last_state(self):
        """
        initialize last_state, which should be called at the start of each epoch
        """
        self.last_state.data.zero_()

    def last_state_detach(self):
        self.last_state.detach_()

    def get_last_state(self,
            node:torch.Tensor|None=None
        ):
        if node is None:
            return self.last_state
        else:
            return self.last_state[node]

    def update_last_state(self,
            node:torch.Tensor,
            pred_state:torch.Tensor,
            label_state:torch.Tensor,
            purpose:Literal["train","test"]=f"test",
            p:float=0.5
        ):
        """
        purpose = train인 경우 p 비율로 Teacher Forcing 적용.
        purpose = test인 경우 그대로 업데이트.

        Input:
            node: [N,]
            pred_state: [N,]
            label_state: [N,]
        """
        if purpose=="train":
            teacher_mask=torch.rand(pred_state.shape,device=pred_state.device)<p
            pred_state=torch.where(teacher_mask,label_state,pred_state)
        self.last_state[node]=pred_state
