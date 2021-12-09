import torch
import warnings
from torch import nn


class TextKernelLoss(nn.Module):
    def __init__(self, ohem_ratio: float = 3, smooth: float = 1e-6):
        super(TextKernelLoss, self).__init__()
        self.smooth = smooth
        self.ohem_ratio = ohem_ratio

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, effective_maps: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            preds (Tensor): The output tensor of size :math:`(N, 6, H, W)`.
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
        selected_masks = ((pred_texts > 0.5).float() * (effective_maps > 0.5).float()).to(preds.device)
        kernel_loss = self.dice_loss(pred_kernels, target_kernels, selected_masks)

        return text_loss, kernel_loss

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, effective_map: torch.Tensor) -> float:
        target[target > 0.5] = 1
        target[target <= 0.5] = 0

        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        effective_map = effective_map.view(effective_map.shape[0], -1)

        pred, target = pred * effective_map, target * effective_map

        PG = torch.sum(pred * target, dim=1)
        P2 = torch.sum(pred * pred, dim=1)
        G2 = torch.sum(target * target, dim=1)

        dice = (2 * PG + self.smooth) / (P2 + G2 + self.smooth)

        return 1 - dice

    def ohem_image(
        self, pred_text: torch.Tensor, target_text: torch.Tensor, effective_map: torch.Tensor
    ) -> torch.Tensor:
        """Sample the top-k maximal negative samples and all positive samples.
        Args:
            pred_text (Float32 Tensor): The pred text score of size :math:`(H, W)`.
            target_text (Float32 Tensor): The ground truth text mask of size :math:`(H, W)`.
            effective_map (Uint8 Tensor): The effective region mask of size :math:`(H, W)`.
        Returns:
            selected_mask (Bool Tensor): The sampled pixel mask of size :math:`(H, W)`.
        """
        assert isinstance(pred_text, torch.Tensor)
        assert isinstance(target_text, torch.Tensor)
        assert isinstance(effective_map, torch.Tensor)
        assert pred_text.shape == target_text.shape
        assert target_text.shape == effective_map.shape
        assert len(pred_text.shape) == 2

        pos_num = int(
            torch.sum(target_text > 0.5).item() - torch.sum((target_text > 0.5) * (effective_map <= 0.5)).item()
        )
        neg_num = int(torch.sum(target_text <= 0.5).item())
        neg_num = min(pos_num * self.ohem_ratio, neg_num)

        if (pos_num == 0) or (neg_num == 0):
            warnings.warn('pos_num = 0 or neg_num = 0')
            return effective_map.bool()

        neg_score = pred_text[target_text <= 0.5]
        neg_score_sorted = torch.sort(neg_score, descending=True)[0]
        threshold = neg_score_sorted[neg_num - 1]

        selected_mask = (((pred_text >= threshold) + (target_text > 0.5)) > 0) * (effective_map > 0.5)

        return selected_mask

    def ohem_batch(
        self, pred_texts: torch.Tensor, target_texts: torch.Tensor, effective_maps: torch.Tensor
    ) -> torch.Tensor:
        """OHEM sampling for a batch of imgs.
        Args:
            pred_texts (Tensor): The text scores of size :math:`(H, W)`.
            target_texts (Tensor): The gt text masks of size :math:`(H, W)`.
            effective_maps (Tensor): The gt effective mask of size :math:`(H, W)`.
        Returns:
            Tensor: The sampled mask of size :math:`(H, W)`.
        """
        assert isinstance(pred_texts, torch.Tensor)
        assert isinstance(target_texts, torch.Tensor)
        assert isinstance(effective_maps, torch.Tensor)
        assert pred_texts.shape == target_texts.shape
        assert target_texts.shape == effective_maps.shape
        assert len(pred_texts.shape) == 3

        selected_masks = []
        for i in range(pred_texts.shape[0]):
            selected_masks.append(
                self.ohem_image(
                    pred_texts[i], target_texts[i], effective_maps[i]
                )
            )

        selected_masks = torch.stack(selected_masks, dim=0)

        return selected_masks
