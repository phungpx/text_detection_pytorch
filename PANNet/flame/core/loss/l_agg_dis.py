import torch
import itertools
from torch import nn


class AggregationDiscriminationLoss(nn.Module):
    def __init__(self, delta_agg: float = 0.5, delta_dis: float = 3):
        super(AggregationDiscriminationLoss, self).__init__()
        self.delta_agg = delta_agg  # is a constant, which is set to 0.5 experimentally and used to filter easy samples.
        self.delta_dis = delta_dis  # is a constant, which is set to 3. experimentally.

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            preds: Tensor, N x 6 x H x W
            targets: Tensor, N x 2 x H x W
        Returns:
            aggregation_loss: float
        '''
        num_samples = preds.shape[0]

        similarity_vectors = preds[:, 2:, :, :]  # N x 4 x W x H
        similarity_vectors = similarity_vectors.reshape(num_samples, 4, -1)  # N x 4 x (H * W)

        target_texts = targets[:, 0, :, :]
        target_texts = target_texts.reshape(num_samples, -1)  # N x (H * W)

        target_kernels = targets[:, 1, :, :]
        target_kernels = target_kernels.reshape(num_samples, -1)  # N x (H * W)

        agg_losses, dis_losses = [], []
        for id_sample in range(num_samples):
            similarity_vector = similarity_vectors[id_sample]
            target_text = target_texts[id_sample]
            target_kernel = target_kernels[id_sample]

            # get the number of text instances
            num_texts = int(target_text.max().item() + 1)   # including background

            instance_agg_losses = []
            G_Ks = []

            # for each text instance (excluding background)
            for id_text in range(1, num_texts):  # all text instances excluding background
                # get ith kernel Ki and ith text instance Ti of ground truth
                Ti = target_text == id_text
                Ki = target_kernel == id_text

                # ignore text instance with no labels
                if (Ki.sum() == 0) or (Ti.sum() == 0):
                    continue

                # get the similarty vector of the ith kernel Ki -> G(Ki) = (sum q of Ki) (F(q)) / |Ki|
                G_Ki = similarity_vector[:, Ki].mean(dim=1)  # 4
                G_Ks.append(G_Ki)  # save for calculation discrimination loss

                # get the similarity vector of all predicted pixels F(p) in text instance Ti
                F_p = similarity_vector[:, Ti]  # 4 x the nums of text pixel

                # compute distance: (||F(p) - G(K_i)|| - delta_agg)
                d = torch.linalg.norm((F_p - G_Ki.reshape(4, 1)), ord=2, dim=0) - self.delta_agg  # the nums of text pixel

                # compute the distance between text pixel p and the kernel Ki of text instance Ti, D(p, K_i)
                D_p_Ki = torch.maximum(d, torch.zeros_like(d)).pow(2)  # the nums of text pixel

                # compute aggregation loss (1 / N) * sum((1 / T) * sum(ln(D(p, Ki)) + 1))
                agg_loss = torch.log(D_p_Ki + 1)  # the nums of text pixel
                agg_loss = agg_loss.mean()

                instance_agg_losses.append(agg_loss)

            if not len(instance_agg_losses):
                instance_agg_loss = torch.tensor(0, device=preds.device, dtype=torch.float)
            else:
                instance_agg_loss = torch.stack(instance_agg_losses).mean()
            agg_losses.append(instance_agg_loss)

            # discrimination loss
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

        return torch.stack(agg_losses), torch.stack(dis_losses)
