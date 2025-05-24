# ──────────────────────────────────────────────────────────────
#  lsfm/models/mmd.py     (수정된 전체 파일)
# ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

def compute_pairwise_distances(x, y):
    # Computes squared Euclidean distances between each pair of rows in x and y
    x_norm = (x**2).sum(dim=1).unsqueeze(1)  # (n, 1)
    y_norm = (y**2).sum(dim=1).unsqueeze(0)  # (1, m)
    dist   = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, min=0.0)


def gaussian_kernel_matrix(x, y, sigmas):
    # Multi-kernel Gaussian: k(x,y) = 1/|sigmas| * sum_u exp(-||x-y||^2/(2*sigma_u^2))
    dist  = compute_pairwise_distances(x, y)  # (n, m)
    betas = 1.0 / (2.0 * torch.tensor(sigmas, device=x.device)
                            .unsqueeze(1).unsqueeze(2))  # (u,1,1)
    expanded = dist.unsqueeze(0) * betas            # (u, n, m)
    kernels  = torch.exp(-expanded)                 # (u, n, m)
    return kernels.mean(dim=0)                      # (n, m)


class MKMMDLoss(nn.Module):
    """
    Multi-Kernel Maximum Mean Discrepancy Loss
    Implements Eq. (10)-(12) from Ruan et al. (LSFM) using multiple Gaussian kernels.

    Args:
        kernels    (list of float, optional): bandwidth sigma values.
                                             Defaults to [2,5,10,20,40].
        layer_ids  (list of int,   optional): 이후에 레이어 선택용으로 사용할 인덱스.
                                             당장은 내부에서 사용하지 않아도 됩니다.
    """
    def __init__(self, kernels=None, layer_ids=None):
        super().__init__()
        # 사용할 σ들
        self.sigmas    = kernels or [2, 5, 10, 20, 40]
        # 나중에 레이어마다 MMD 계산을 모아두고 싶을 때를 위해 저장
        self.layer_ids = layer_ids

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # source, target 은 (batch, feature_dim) 형태의 텐서여야 합니다.
        K_xx = gaussian_kernel_matrix(source, source, self.sigmas)
        K_yy = gaussian_kernel_matrix(target, target, self.sigmas)
        K_xy = gaussian_kernel_matrix(source, target, self.sigmas)
        # MMD^2 = E[K_xx] + E[K_yy] - 2 E[K_xy]
        return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()


if __name__ == '__main__':
    # Quick sanity check
    source = torch.randn(5, 128)
    target = torch.randn(7, 128)
    loss = MKMMDLoss(layer_ids=[2,4,6])  # 이제 에러 없이 받습니다.
    print(loss(source, target))  # scalar tensor
