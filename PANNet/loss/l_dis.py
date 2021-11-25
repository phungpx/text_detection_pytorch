import torch
import itertools
from torch import nn


class DistanceLoss(nn.Module):
    def __init__(self, delta_dis: float = 3):
        super(DistanceLoss, self).__init__()
        self.delta_dis = delta_dis  # is a constant, which is set to 3. experimentally.

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            preds: Tensor, N x 6 x H / 4 x W / 4  (stride 4 compare to input size)
            targets: Tensor, N x 2 x H / 4 x W / 4
        Output:
            distance_loss: float
        '''
        num_samples = preds.shape[0]

        similarity_vectors = preds[:, 2:, :, :]  # N x 4 x W' x H'
        similarity_vectors = similarity_vectors.reshape(num_samples, 4, -1)  # N x 4 x (H' * W')

        target_texts = targets[:, 0, :, :]
        target_texts = target_texts.reshape(num_samples, -1)  # N x (H' * W')

        target_kernels = targets[:, 1, :, :]
        target_kernels = target_kernels.reshape(num_samples, -1)  # N x (H' * W')

        dis_losses = []
        for id_sample in range(num_samples):
            similarity_vector = similarity_vectors[id_sample]
            target_text = target_texts[id_sample]
            target_kernel = target_kernels[id_sample]

            # get the number of text instances
            num_texts = int(target_text.max().item() + 1)   # including background

            G_Ks = []
            for id_text in range(1, num_texts):  # all text instances excluding background
                # get ith kernel Ki and ith text instance Ti of ground truth
                Ti = target_text == id_text
                Ki = target_kernel == id_text

                # ignore text instance with no labels
                if (Ki.sum() == 0) or (Ti.sum() == 0):
                    continue

                # get the similarty vector of the ith kernel Ki -> G(Ki) = (sum q of Ki) (F(q)) / |Ki|
                G_Ki = similarity_vector[:, Ki].mean(dim=1)  # 4
                G_Ks.append(G_Ki)

            dis_loss = 0
            for G_Ki, G_Kj in itertools.combinations(G_Ks, 2):
                # compute D(Ki, Kj) = max(delta_dis - ||G(K_i) - G(K_j)||, 0) ** 2
                D_Ki_Kj = self.delta_dis - torch.linalg.norm((G_Ki - G_Kj), ord=2)  # 1
                D_Ki_Kj = torch.maximum(D_Ki_Kj, torch.zeros_like(D_Ki_Kj)).pow(2)  # 1
                # sum_N(sum_N(ln(D(Ki, Kj)) + 1))
                dis_loss += torch.log(D_Ki_Kj + 1)

            if len(G_Ks) > 1:
                N = len(G_Ks)
                dis_loss = dis_loss / (N * (N - 1))
            else:
                dis_loss = torch.tensor(0, device=preds.device, dtype=torch.float)

            dis_losses.append(dis_loss)

        return torch.stack(dis_losses)
