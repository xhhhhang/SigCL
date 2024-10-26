import torch

from .base import SigCLossBase

# class SigCLossPN(SigCLossBase):
#     def _loss(self, loss_info, extra_info, first_features, **kwargs):
#         loss_matrix = loss_info["loss_matrix"]
#         pos_mask = loss_info["pos_mask"]
#         neg_mask = loss_info["neg_mask"]
#         num_pos = extra_info["num_pos"]
#         num_neg = extra_info["num_neg"]

#         pos_loss = (loss_matrix * pos_mask).sum() / num_pos
#         neg_loss = (loss_matrix * neg_mask).sum() / num_neg
#         loss = pos_loss + first_features.shape[0] * neg_loss

#         return loss


class SigCLossNegWeight(SigCLossBase):
    def __init__(self, *args, max_neg_weight=16, neg_weight_step=1.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_weight = 1
        self.max_neg_weight = max_neg_weight
        self.neg_weight_step = neg_weight_step

    def _loss(
        self,
        loss_info,
        extra_info,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
    ):
        loss_matrix = loss_info["loss_matrix"]
        pos_mask = loss_info["pos_mask"]
        neg_mask = loss_info["neg_mask"]
        num_pos = extra_info["num_pos"]
        num_neg = extra_info["num_neg"]

        pos_loss = (loss_matrix * pos_mask).sum() / num_pos
        neg_loss = (loss_matrix * neg_mask).sum() / num_neg
        loss = pos_loss + self.neg_weight * neg_loss

        return loss

    def set_neg_weight(self, neg_weight):
        self.neg_weight = neg_weight

    def step_neg_weight(self):
        if self.neg_weight >= self.max_neg_weight:
            return
        self.neg_weight *= self.neg_weight_step
        if self.neg_weight > self.max_neg_weight:
            self.neg_weight = self.max_neg_weight


class SigCLossAverage(SigCLossBase):
    def _aggregate_loss(self, loss_dict):
        return (loss_dict["pos_loss_sum"] + loss_dict["neg_loss_sum"]) / (
            loss_dict["num_pos"] + loss_dict["num_neg"]
        )


class SigCLossRatio(SigCLossBase):
    def _aggregate_loss(self, loss_dict):
        return (loss_dict["pos_loss_sum"] + self.neg_weight * loss_dict["neg_loss_sum"]) / (
            loss_dict["num_pos"] + loss_dict["num_neg"] * self.neg_weight
        )


class SigCLossAverageV2(SigCLossBase):
    def _aggregate_loss(self, loss_dict):
        return (loss_dict["pos_loss_sum"] + loss_dict["neg_loss_sum"]) / loss_dict["num_pos"]


class FocalBase(SigCLossBase):
    def __init__(self, gamma=2.0, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.normalize = normalize

    def _loss(
        self,
        loss_info,
        extra_info,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
    ):
        logits = loss_info["logits"]
        pos_mask = loss_info["pos_mask"]
        neg_mask = loss_info["neg_mask"]
        labels = loss_info["labels"]
        loss_matrix = loss_info["loss_matrix"]

        p = torch.sigmoid(logits * labels)
        focal_weight = (1 - p) ** self.gamma
        loss_matrix = loss_matrix * focal_weight

        pos_loss_sum = (loss_matrix * pos_mask).sum()
        neg_loss_sum = (loss_matrix * neg_mask).sum()
        num_pos = pos_mask.sum().clamp(min=1)
        num_neg = neg_mask.sum().clamp(min=1)

        if self.normalize:
            pos_loss_sum *= 1 + self.gamma
            neg_loss_sum *= 1 + self.gamma

        return {
            "pos_loss_sum": pos_loss_sum,
            "neg_loss_sum": neg_loss_sum,
            "num_pos": num_pos,
            "num_neg": num_neg,
        }


class FocalAverage(FocalBase, SigCLossAverageV2):
    pass

class ExpBase(SigCLossBase):
    def _loss(
        self,
        loss_info,
        extra_info,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
    ):
        logits = loss_info["logits"]
        pos_mask = loss_info["pos_mask"]
        neg_mask = loss_info["neg_mask"]
        labels = loss_info["labels"]
        loss_matrix = loss_info["loss_matrix"]

        p = torch.sigmoid(logits * labels)

        # self.neg_weight is the base of the exponent, not real NEG weight, just for convenience
        exp_weight = self.neg_weight ** (1 - p)
        loss_matrix = loss_matrix * exp_weight

        pos_loss_sum = (loss_matrix * pos_mask).sum()
        neg_loss_sum = (loss_matrix * neg_mask).sum()
        num_pos = pos_mask.sum().clamp(min=1)
        num_neg = neg_mask.sum().clamp(min=1)

        return {
            "pos_loss_sum": pos_loss_sum,
            "neg_loss_sum": neg_loss_sum,
            "num_pos": num_pos,
            "num_neg": num_neg,
        }

class ExpAverage(ExpBase, SigCLossAverageV2):
    pass
