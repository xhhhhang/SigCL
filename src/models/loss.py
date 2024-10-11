import torch
import torch.nn as nn
from torch.nn import functional as F


class SigCLoss(nn.Module):
    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
        bidir=True,
        use_horovod=False,
        mask_diagonal=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.bidir = bidir
        self.mask_diagonal = mask_diagonal

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
    ):
        logits = self.get_logits(first_features, second_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            first_features.device,
            first_features.dtype,
            first_label,
            second_label,
        )
        loss_matrix = -F.logsigmoid(labels * logits)
        if self.mask_diagonal:
            loss_matrix.fill_diagonal_(0)
        loss = loss_matrix.sum() / first_features.shape[0]
        return loss

    def _loss_v2(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias=None,
    ):
        logits = self.get_logits(first_features, second_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            first_features.device,
            first_features.dtype,
            first_label,
            second_label,
        )
        loss_matrix = -F.logsigmoid(labels * logits)
        if self.mask_diagonal:
            loss_matrix.fill_diagonal_(0)

        pos_mask = labels == 1
        neg_mask = labels == -1
        num_pos = pos_mask.sum().clamp(min=1)
        num_neg = neg_mask.sum().clamp(min=1)

        pos_loss = (loss_matrix * pos_mask).sum() / num_pos
        neg_loss = (loss_matrix * neg_mask).sum() / num_neg * (first_features.shape[0] - 1)

        loss = pos_loss + neg_loss
        return loss

    def forward(
        self,
        first_features,
        second_features,
        first_label,
        second_label,
        logit_scale,
        logit_bias,
        output_dict=False,
    ):
        loss = self._loss_v2(
            first_features, second_features, first_label, second_label, logit_scale, logit_bias
        )
        return {"contrastive_loss": loss} if output_dict else loss


#         if self.world_size > 1:
#             # exchange second features w/ neighbour world_size - 1 times
#             right_rank = (self.rank + 1) % self.world_size
#             left_rank = (self.rank - 1 + self.world_size) % self.world_size
#             if self.bidir:
#                 second_features_to_right = second_features_to_left = second_features
#                 second_label_to_right = second_label_to_left = second_label
#                 num_bidir, remainder = divmod(self.world_size - 1, 2)
#                 for i in range(num_bidir):
#                     second_features_recv, second_label_recv = neighbour_exchange_bidir_with_grad(
#                         left_rank,
#                         right_rank,
#                         (second_features_to_left, second_label_to_left),
#                         (second_features_to_right, second_label_to_right),
#                     )

#                     for f, l in zip(second_features_recv, second_label_recv):
#                         loss += self._loss(
#                             first_features,
#                             f,
#                             first_label,
#                             l,
#                             logit_scale,
#                             logit_bias,
#                             mask_diagonal,
#                         )
#                     second_features_to_left, second_label_to_left = second_features_recv[0], second_label_recv[0]
#                     second_features_to_right, second_label_to_right = second_features_recv[1], second_label_recv[1]

#                 if remainder:
#                     second_features_recv, second_label_recv = neighbour_exchange_with_grad(
#                         left_rank, right_rank, (second_features_to_right, second_label_to_right))

#                     loss += self._loss(
#                         first_features,
#                         second_features_recv,
#                         first_label,
#                         second_label_recv,
#                         logit_scale,
#                         logit_bias,
#                         mask_diagonal,
#                     )
#             else:
#                 second_features_to_right = second_features
#                 second_label_to_right = second_label
#                 for i in range(self.world_size - 1):
#                     second_features_from_left, second_label_from_left = neighbour_exchange_with_grad(
#                         left_rank, right_rank, (second_features_to_right, second_label_to_right))

#                     loss += self._loss(
#                         first_features,
#                         second_features_from_left,
#                         first_label,
#                         second_label_from_left,
#                         logit_scale,
#                         logit_bias,
#                         mask_diagonal,
#                     )
#                     second_features_to_right = second_features_from_left
#                     second_label_to_right = second_label_from_left

# return {"contrastive_loss": loss} if output_dict else loss

# # Helper functions for distributed operations (you may need to implement these)
# def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
#     # Implement this function for bidirectional exchange
#     pass

# def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
#     # Implement this function for unidirectional exchange
#     pass
