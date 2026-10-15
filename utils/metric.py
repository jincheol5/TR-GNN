import torch

class Metric:
    @staticmethod
    def compute_accuracy(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Input:
            pred_logit
            label
        Return:
            acc
        """
        pred=(torch.sigmoid(pred_logit)>=0.5).float()
        acc=(pred==label.float()).float().mean().item()
        return acc

    @staticmethod
    def compute_hop_range_accuracy(
            pred_logit:torch.Tensor,
            label:torch.Tensor,
            hop:torch.Tensor
        ):
        """
        hop_range_1: 1<=hop<3
        hop_range_2: 3<=hop<5
        hop_range_3: 5<=hop

        Input:
            pred_logit: [B,]
            label: [B,]
            hop: [B,]
        Return:
            Dictionary containing overall accuracy and positive-only accuracy
            for each hop range. Empty ranges have accuracy 0.0.
        """
        pred=(torch.sigmoid(pred_logit)>=0.5).float()
        correct=(pred==label.float())
        positive_mask=(label==1)
        hop_range_masks=(
            positive_mask & (hop>=1) & (hop<3),
            positive_mask & (hop>=3) & (hop<5),
            positive_mask & (hop>=5)
        )
        result={"acc":correct.float().mean().item()}
        for range_idx,mask in enumerate(hop_range_masks,start=1):
            result[f"hop_range_{range_idx}_acc"]=(
                correct[mask].float().mean().item() if mask.any() else 0.0
            )
        return result

    # @staticmethod
    # def compute_hop_range_accuracy_old(
    #         pred_logit:torch.Tensor,
    #         label:torch.Tensor,
    #         hop_range_1_mask:torch.Tensor,
    #         hop_range_2_mask:torch.Tensor,
    #         hop_range_3_mask:torch.Tensor
    #     ):
    #     """
    #     Input:
    #         pred_logit
    #         label
    #         pos_mask
    #         hop_range_1_mask
    #         hop_range_2_mask
    #         hop_range_3_mask
    #     Return:
    #         acc
    #         hop_range_1_acc
    #         hop_range_2_acc
    #         hop_range_3_acc
    #     """
    #     pred=(torch.sigmoid(pred_logit)>=0.5).float()
    #     acc=(pred==label.float()).float().mean().item()

    #     ### range 1 accuracy
    #     if hop_range_1_mask.any():
    #         hop_range_1_acc=(pred[hop_range_1_mask]==label[hop_range_1_mask]).float().mean().item()
    #     else:
    #         hop_range_1_acc=0.0

    #     ### range 2 accuracy
    #     if hop_range_2_mask.any():
    #         hop_range_2_acc=(pred[hop_range_2_mask]==label[hop_range_2_mask]).float().mean().item()
    #     else:
    #         hop_range_2_acc=0.0

    #     ### range 3 accuracy
    #     if hop_range_3_mask.any():
    #         hop_range_3_acc=(pred[hop_range_3_mask]==label[hop_range_3_mask]).float().mean().item()
    #     else:
    #         hop_range_3_acc=0.0
    #     return {
    #         "acc":acc,
    #         "hop_range_1_acc":hop_range_1_acc,
    #         "hop_range_2_acc":hop_range_2_acc,
    #         "hop_range_3_acc":hop_range_3_acc
    #     }
