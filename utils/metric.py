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

    @staticmethod
    def compute_PPR(
            pred_logit:torch.Tensor
        ):
        """
        PPR (Prediction Positive Rate)
        -> 전체 예측 중 Positive로 분류한 비율

        0%: 모든 샘플을 Negative로 예측
        20%: Negative 예측에 치우침
        50%: Positive와 Negative를 같은 비율로 예측
        80%: Positive 예측에 치우침
        100%: 모든 샘플을 Positive로 예측
        """
        # Return a fraction in [0, 1], consistent with compute_accuracy.
        pred=(torch.sigmoid(pred_logit)>=0.5).float()
        return pred.mean().item()

    @staticmethod
    def compute_Prediction_Bias(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Prediction Bias
        -> 실제 Positive 비율과 예측 Positive 비율의 차이를 계산

        Bias = 0: 예측 클래스 비율이 실제 비율과 일치
        Bias > 0: Positive를 과도하게 예측
        Bias < 0: Negative를 과도하게 예측
        """
        # Signed difference in positive rates, in [-1, 1].
        predicted_positive_rate=Metric.compute_PPR(pred_logit=pred_logit)
        actual_positive_rate=(label==1).float().mean().item()
        return predicted_positive_rate-actual_positive_rate
