import torch
from torch import nn
from typing import Callable

from .l_dis import DistanceLoss
from .l_agg import AggregationLoss
from .l_tex_ker import TextKernelLoss


class PANLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        delta_agg: float = 0.5,
        delta_dis: float = 3,
        ohem_ratio: float = 3,
        reduction: str = 'mean',
    ):
        super(PANLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.dis_loss_fn = DistanceLoss(delta_dis=delta_dis)
        self.agg_loss_fn = AggregationLoss(delta_agg=delta_agg)
        self.text_kernel_loss_fn = TextKernelLoss(ohem_ratio=ohem_ratio)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, effective_maps: torch.Tensor):
        agg_loss = self.agg_loss_fn(preds, targets)
        dis_loss = self.dis_loss_fn(preds, targets)
        text_loss, kernel_loss = self.text_kernel_loss_fn(preds, targets, effective_maps)

        if self.reduction == 'mean':
            agg_loss = agg_loss.mean()
            dis_loss = dis_loss.mean()
            text_loss = text_loss.mean()
            kernel_loss = kernel_loss.mean()

        elif self.reduction == 'sum':
            agg_loss = agg_loss.sum()
            dis_loss = dis_loss.sum()
            text_loss = text_loss.sum()
            kernel_loss = kernel_loss.sum()

        else:
            raise NotImplementedError

        loss = text_loss + self.alpha * kernel_loss + self.beta * (agg_loss + dis_loss)

        return loss, agg_loss, dis_loss, text_loss, kernel_loss
