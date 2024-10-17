from .base import SigCLossBase


class SigCLossPN(SigCLossBase):
    def _loss(self, loss_info, extra_info, first_features, **kwargs):
        loss_matrix = loss_info["loss_matrix"]
        pos_mask = loss_info["pos_mask"]
        neg_mask = loss_info["neg_mask"]
        num_pos = extra_info["num_pos"]
        num_neg = extra_info["num_neg"]

        pos_loss = (loss_matrix * pos_mask).sum() / num_pos
        neg_loss = (loss_matrix * neg_mask).sum() / num_neg
        loss = pos_loss + first_features.shape[0] * neg_loss

        return loss


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
