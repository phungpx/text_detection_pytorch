import torch
from torch import nn


class TextKernelLoss(nn.Module):
    def __init__(self, ohem_ratio: float = 3, epsilon: float = 1e-6):
        super(TextKernelLoss, self).__init__()
        self.epsilon = epsilon
        self.ohem_ratio = ohem_ratio

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, effective_maps: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            preds: Tensor, N x 6 x H x W
            targets: Tensor, N x 2 x H x W
            effective_maps: Tensor, N x H x W
        Returns:
            distance_loss: float
        '''
        pred_texts = torch.sigmoid(preds[:, 0, :, :])
        pred_kernels = torch.sigmoid(preds[:, 1, :, :])

        target_texts = targets[:, 0, :, :]
        target_kernels = targets[:, 1, :, :]

        # text region loss
        selected_masks = self.ohem_batch(pred_texts, target_texts, effective_maps).to(preds.device)
        text_loss = self.dice_loss(pred_texts, target_texts, selected_masks)

        # kernel loss
        selected_masks = ((pred_texts > 0.5) & (effective_maps > 0.5)).float().to(preds.device)
        kernel_loss = self.dice_loss(pred_kernels, target_kernels, selected_masks)

        return text_loss, kernel_loss

    def dice_loss(self, pred, target, effective_map):
        target[target > 0.5] = 1
        target[target <= 0.5] = 0

        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        effective_map = effective_map.view(effective_map.shape[0], -1)

        pred, target = pred * effective_map, target * effective_map

        PG = torch.sum(pred * target, dim=1)
        P2 = torch.sum(pred ** 2, dim=1)
        G2 = torch.sum(target ** 2, dim=1)

        dice = (2 * PG) / (P2 + G2 + self.epsilon)

        return 1 - dice

    def ohem(self, pred_text, target_text, effective_map):
        positive_number = torch.sum(target_text > 0.5) - torch.sum((target_text > 0.5) & (effective_map <= 0.5))
        if positive_number.item() == 0:
            return effective_map.float()

        negative_number = torch.sum(target_text <= 0.5)
        negative_number = torch.min(positive_number * self.ohem_ratio, negative_number)
        if negative_number.item() == 0:
            return effective_map.float()

        negative_scores = pred_text[target_text <= 0.5]
        negative_scores = torch.sort(negative_scores, descending=True)[0]
        score_threshold = negative_scores[negative_number - 1]

        selected_mask = ((pred_text >= score_threshold) | (target_text > 0.5)) & (effective_map > 0.5)

        return selected_mask.float()

    def ohem_batch(self, pred_texts, target_texts, effective_maps):
        selected_masks = []
        for i in range(pred_texts.shape[0]):
            selected_masks.append(
                self.ohem(pred_texts[i, :, :], target_texts[i, :, :], effective_maps[i, :, :])
            )

        selected_masks = torch.stack(selected_masks, dim=0)

        return selected_masks
