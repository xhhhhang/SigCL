import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from open_clip.loss import (
    neighbour_exchange_bidir_with_grad,
    neighbour_exchange_with_grad,
)


class SigCLossBase(nn.Module):
    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
        bidir=True,
        use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, first_label, second_label):
        labels = -torch.ones((len(first_label), len(second_label)), device=device, dtype=dtype)
        labels += 2 * torch.eq(first_label.unsqueeze(1), second_label.unsqueeze(0)).float()

        return labels

    def get_logits(self, first_features, second_features, logit_scale, logit_bias=None):
        logits = logit_scale * first_features @ second_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
    ):
        logits = self.get_logits(first_features, second_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            first_features.device,
            first_features.dtype,
            first_label,
            second_label,
        )
        loss_matrix = -F.logsigmoid(labels * logits)
        if mask_diagonal:
            loss_matrix.fill_diagonal_(0)
        loss = loss_matrix.sum() / first_features.shape[0]
        return loss

    def forward(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        mask_diagonal=False,
        output_dict=False,
        **kwargs,
    ):
        loss = self._loss(
            first_features,
            second_features,
            first_label,
            second_label,
            mask_diagonal=mask_diagonal,
            **kwargs,
        )

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                second_features_to_right = second_features_to_left = second_features
                second_label_to_right = second_label_to_left = second_label
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    second_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        second_features_to_left,
                        second_features_to_right,
                    )
                    second_label_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        second_label_to_left,
                        second_label_to_right,
                    )

                    for f_recv, l_recv in zip(second_features_recv, second_label_recv):
                        loss += self._loss(
                            first_features,
                            f_recv,
                            first_label,
                            l_recv,
                            mask_diagonal=mask_diagonal,
                            **kwargs,
                        )
                    second_features_to_left, second_features_to_right = second_features_recv
                    second_label_to_left, second_label_to_right = second_label_recv

                if remainder:
                    second_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, second_features_to_right
                    )
                    second_label_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, second_label_to_right
                    )

                    loss += self._loss(
                        first_features,
                        second_features_recv,
                        first_label,
                        second_label_recv,
                        mask_diagonal=mask_diagonal,
                        **kwargs,
                    )
            else:
                second_features_to_right = second_features
                second_label_to_right = second_label
                for i in range(self.world_size - 1):
                    second_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, second_features_to_right
                    )
                    second_label_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, second_label_to_right
                    )

                    loss += self._loss(
                        first_features,
                        second_features_from_left,
                        first_label,
                        second_label_from_left,
                        mask_diagonal=mask_diagonal,
                        **kwargs,
                    )
                    second_features_to_right = second_features_from_left
                    second_label_to_right = second_label_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class SigCLossNegHard(SigCLossBase):
    def __init__(self, *args, min_neg_samples=1, neg_weight_step=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_neg_samples = min_neg_samples
        self.neg_weight_step = neg_weight_step
        self.neg_weight = 0

    def _loss(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
    ):
        logits = self.get_logits(first_features, second_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            first_features.device,
            first_features.dtype,
            first_label,
            second_label,
        )
        loss_matrix = -F.logsigmoid(labels * logits)
        if mask_diagonal:
            loss_matrix.fill_diagonal_(0)

        pos_mask = labels == 1
        neg_mask = labels == -1
        num_pos = pos_mask.sum().clamp(min=1)
        num_neg = neg_mask.sum().clamp(min=1)

        # Calculate positive loss
        pos_loss = (loss_matrix * pos_mask).sum()

        # Select hard negatives
        num_hard_neg = max(num_pos + self.neg_weight, self.min_neg_samples)
        neg_losses = loss_matrix[neg_mask]
        if num_hard_neg < num_neg:
            hard_neg_losses, _ = torch.topk(neg_losses, int(num_hard_neg), largest=True)
        else:
            hard_neg_losses = neg_losses

        # print(f'pos:{num_pos}, neg: {min(num_hard_neg, num_neg)}')
        # Calculate negative loss
        neg_loss = hard_neg_losses.sum()

        loss = (pos_loss + neg_loss) / num_pos
        return loss

    def set_neg_weight(self, neg_weight):
        self.neg_weight = neg_weight

    def step_neg_weight(self):
        self.neg_weight += self.neg_weight_step
        # Prevent overflow by clamping to the maximum value of float32
        self.neg_weight = min(self.neg_weight, torch.finfo(torch.float32).max)


class SigCLossNegWeight(SigCLossBase):
    def __init__(self, *args, max_neg_weight=16, neg_weight_step=1.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_weight = 1
        self.max_neg_weight = max_neg_weight
        self.neg_weight_step = neg_weight_step

    def _loss(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
    ):
        logits = self.get_logits(first_features, second_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            first_features.device,
            first_features.dtype,
            first_label,
            second_label,
        )
        loss_matrix = -F.logsigmoid(labels * logits)
        if mask_diagonal:
            loss_matrix.fill_diagonal_(0)

        pos_mask = labels == 1
        neg_mask = labels == -1
        num_pos = pos_mask.sum().clamp(min=1)
        num_neg = neg_mask.sum().clamp(min=1)

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


class SigCLossPN(SigCLossBase):
    def _loss(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        pos_logit_scale,
        neg_logit_scale,
        pos_logit_bias=None,
        neg_logit_bias=None,
        mask_diagonal=False,
    ):
        labels = self.get_ground_truth(
            first_features.device, first_features.dtype, first_label, second_label
        )

        pos_logits = self.get_logits(
            first_features, second_features, pos_logit_scale, pos_logit_bias
        )
        pos_loss_matrix = -F.logsigmoid(pos_logits)
        neg_logits = self.get_logits(
            first_features, second_features, neg_logit_scale, neg_logit_bias
        )
        neg_loss_matrix = -F.logsigmoid(neg_logits)

        pos_mask = labels == 1
        if mask_diagonal:
            pos_mask.fill_diagonal_(0)
        neg_mask = labels == -1

        pos_loss = (pos_loss_matrix * pos_mask).sum() / pos_mask.sum().clamp(min=1)
        neg_loss = (neg_loss_matrix * neg_mask).sum() / neg_mask.sum().clamp(min=1)

        return pos_loss + neg_loss * first_features.shape[0]
