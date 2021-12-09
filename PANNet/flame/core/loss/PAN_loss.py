import torch
from typing import Callable

from . import loss
from .l_tex_ker import TextKernelLoss
from .l_agg_dis import AggregationDiscriminationLoss


class PANLoss(loss.LossBase):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        delta_agg: float = 0.5,
        delta_dis: float = 3,
        ohem_ratio: float = 3,
        reduction: str = 'mean',
        output_transform: Callable = lambda x: x,
    ):
        super(PANLoss, self).__init__(output_transform)
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.text_kernel_loss_fn = TextKernelLoss(ohem_ratio=ohem_ratio)
        self.agg_dis_loss_fn = AggregationDiscriminationLoss(delta_agg=delta_agg, delta_dis=delta_dis)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, effective_maps: torch.Tensor):
        agg_loss, dis_loss = self.agg_dis_loss_fn(preds, targets)
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

        return loss
