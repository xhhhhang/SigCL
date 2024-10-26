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
        bidir=True,
        neg_weight=1,
        fabric=None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.fabric = fabric
        self.rank = fabric.local_rank
        self.world_size = fabric.world_size
        self.bidir = bidir

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.neg_weight = neg_weight

    def get_ground_truth(self, device, dtype, first_label, second_label):
        labels = -torch.ones((len(first_label), len(second_label)), device=device, dtype=dtype)
        labels += 2 * torch.eq(first_label.unsqueeze(1), second_label.unsqueeze(0)).float()

        return labels

    def get_logits(self, first_features, second_features, logit_scale, logit_bias=None):
        logits = logit_scale * first_features @ second_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss_helper(
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
        if mask_diagonal:
            labels.fill_diagonal_(0)
        pos_mask = labels == 1
        neg_mask = labels == -1
        num_pos = pos_mask.sum().clamp(min=1)
        num_neg = neg_mask.sum().clamp(min=1)

        average_pos_logits = (logits * pos_mask).sum() / num_pos
        average_neg_logits = (logits * neg_mask).sum() / num_neg

        loss_matrix = -F.logsigmoid(labels * logits)
        loss_info = {
            "loss_matrix": loss_matrix,
            "labels": labels,
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
        }
        extra_info = {
            "num_pos": num_pos,
            "num_neg": num_neg,
            "average_pos_logits": average_pos_logits,
            "average_neg_logits": average_neg_logits,
        }
        return loss_info, extra_info

    def _aggregate_loss(self, loss_dict):
        return (
            loss_dict["pos_loss_sum"] / loss_dict["num_pos"]
            + loss_dict["neg_loss_sum"] / loss_dict["num_neg"] * self.neg_weight
        )

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

        pos_loss_sum = (loss_matrix * pos_mask).sum()
        neg_loss_sum = (loss_matrix * neg_mask).sum()
        return {
            "pos_loss_sum": pos_loss_sum,
            "neg_loss_sum": neg_loss_sum,
            "num_pos": num_pos,
            "num_neg": num_neg,
        }

    @staticmethod
    def sum_loss_dict(first_loss_dict, second_loss_dict):
        return {
            "pos_loss_sum": first_loss_dict["pos_loss_sum"] + second_loss_dict["pos_loss_sum"],
            "neg_loss_sum": first_loss_dict["neg_loss_sum"] + second_loss_dict["neg_loss_sum"],
            "num_pos": first_loss_dict["num_pos"] + second_loss_dict["num_pos"],
            "num_neg": first_loss_dict["num_neg"] + second_loss_dict["num_neg"],
        }

    def forward(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
        mask_diagonal=False,
        output_dict=False,
        return_extra_info=False,
        **kwargs,
    ):
        loss_info, extra_info = self._loss_helper(
            first_features=first_features,
            second_features=second_features,
            first_label=first_label,
            second_label=second_label,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            mask_diagonal=mask_diagonal,
        )

        if self.world_size <= 1:
            loss_dict = self._loss(
                loss_info=loss_info,
                extra_info=extra_info,
                first_features=first_features,
                second_features=second_features,
                first_label=first_label,
                second_label=second_label,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                mask_diagonal=mask_diagonal,
                **kwargs,
            )
            loss = self._aggregate_loss(loss_dict)
            return loss if not output_dict else {**extra_info, "loss": loss}

        else:
            loss_dict = self._loss(
                loss_info=loss_info,
                extra_info=extra_info,
                first_features=first_features,
                second_features=second_features,
                first_label=first_label,
                second_label=second_label,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                mask_diagonal=mask_diagonal,
                **kwargs,
            )
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
                        loss_dict = self.sum_loss_dict(
                            loss_dict,
                            self._loss(
                                loss_info=loss_info,
                                extra_info=extra_info,
                                first_features=first_features,
                                second_features=f_recv,
                                first_label=first_label,
                                second_label=l_recv,
                                mask_diagonal=mask_diagonal,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                **kwargs,
                            ),
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

                    loss_dict = self.sum_loss_dict(
                        loss_dict,
                        self._loss(
                            loss_info=loss_info,
                            extra_info=extra_info,
                            first_features=first_features,
                            second_features=second_features_recv,
                            first_label=first_label,
                            second_label=second_label_recv,
                            mask_diagonal=False,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            **kwargs,
                        ),
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

                    loss_dict = self.sum_loss_dict(
                        loss_dict,
                        self._loss(
                            loss_info=loss_info,
                            extra_info=extra_info,
                            first_features=first_features,
                            second_features=second_features_from_left,
                            first_label=first_label,
                            second_label=second_label_from_left,
                            mask_diagonal=False,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            **kwargs,
                        ),
                    )
                    second_features_to_right = second_features_from_left
                    second_label_to_right = second_label_from_left

            # print(f"pos num: {loss_dict['num_pos']}, neg num: {loss_dict['num_neg']}")
            return self._aggregate_loss(loss_dict)
