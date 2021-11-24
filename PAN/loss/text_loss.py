from torch import nn


class TextLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super(TextLoss, self).__init__()
        self.epsilon = epsilon

    def dice_loss(self, pred, target, mask):
        num_samples = pred.shape[0]

        pred = torch.sigmoid(pred)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1

        pred = pred.view(num_samples, -1)
        target = target.view(num_samples, -1)
        mask = mask.view(num_samples, -1)

        pred, target = pred * mask, target * mask

        PG = torch.sum(pred * target, dim=1)
        P2 = torch.sum(pred ** 2, dim=1)
        G2 = torch.sum(target ** 2, dim=1)

        dice = (2 * PG) / (P2 + G2 + self.epsilon)

        return 1 - dice
