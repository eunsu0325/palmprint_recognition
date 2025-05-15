import torch
import torch.nn as nn

def compute_pairwise_distances(x, y):
    # Computes squared Euclidean distances between each pair of rows in x and y
    x_norm = (x**2).sum(dim=1).unsqueeze(1)  # (n, 1)
    y_norm = (y**2).sum(dim=1).unsqueeze(0)  # (1, m)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, min=0.0)


def gaussian_kernel_matrix(x, y, sigmas):
    # Multi-kernel Gaussian: k(x,y) = 1/|sigmas| * sum_u exp(-||x-y||^2/(2*sigma_u^2))
    dist = compute_pairwise_distances(x, y)  # (n, m)
    # Prepare betas: (u, 1, 1)
    betas = 1.0 / (2.0 * torch.tensor(sigmas, device=x.device).unsqueeze(1).unsqueeze(2))  # (u,1,1)
    # Expand dist to (u, n, m) and apply each beta
    expanded = dist.unsqueeze(0) * betas  # (u, n, m)
    kernels = torch.exp(-expanded)         # (u, n, m)
    # Average across kernels
    return kernels.mean(dim=0)            # (n, m)


class MKMMDLoss(nn.Module):
    """
    Multi-Kernel Maximum Mean Discrepancy Loss
    Implements Eq. (10)-(12) from Ruan et al. (LSFM) using multiple Gaussian kernels.

    Args:
        kernels (list of float): bandwidths sigma values. Defaults to [2,5,10,20,40].
    """
    def __init__(self, kernels=None):
        super().__init__()
        # Default bandwidths if none provided
        self.sigmas = kernels or [2, 5, 10, 20, 40]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute kernel matrices
        K_xx = gaussian_kernel_matrix(source, source, self.sigmas)
        K_yy = gaussian_kernel_matrix(target, target, self.sigmas)
        K_xy = gaussian_kernel_matrix(source, target, self.sigmas)
        # MMD^2 = E[K_xx] + E[K_yy] - 2 E[K_xy]
        return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()


if __name__ == '__main__':
    # Quick sanity check
    source = torch.randn(5, 128)
    target = torch.randn(7, 128)
    loss = MKMMDLoss()
    print(loss(source, target))  # scalar tensor
