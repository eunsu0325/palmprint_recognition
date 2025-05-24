# lsfm/models/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import ResBlk

class Discriminator(nn.Module):
    """
    StarGAN v2 기반 Discriminator를 128/256 등
    다양한 해상도에서 동작하도록 수정한 버전.

    Args:
        img_size    (int): 입력 이미지 크기 (128, 256, ...)
        in_channels (int): 입력 채널 수 (1: grayscale)
        num_domains (int): 도메인(뷰) 개수
        max_conv_dim(int): 최대 채널 수 상한선
    """
    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 1,
        num_domains: int = 4,
        max_conv_dim: int = 512
    ):
        super().__init__()

        # StarGAN v2 관습: 첫 conv 채널 = 2^14 / img_size
        dim_in = 2 ** 14 // img_size
        dim = dim_in

        blocks = []
        # 1) from-rgb
        blocks.append(nn.Conv2d(in_channels, dim, 3, 1, 1))

        # 2) repeat down-sample residual blocks
        repeat_num = int(torch.log2(torch.tensor(img_size)).item()) - 2
        for _ in range(repeat_num):
            dim_out = min(dim * 2, max_conv_dim)
            blocks.append(ResBlk(dim, dim_out, downsample=True))
            dim = dim_out

        # 3) LeakyReLU → adaptive pooling → flatten
        self.main = nn.Sequential(*blocks)
        self.activation = nn.LeakyReLU(0.2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))    # (B, dim, 1,1)
        self.fc = nn.Linear(dim, num_domains)      # 최종 도메인 판별

    def forward(self, x: torch.Tensor, y: torch.LongTensor):
        """
        x: (B, C, H, W), y: (B,) 도메인 라벨
        return: (B, )
        """
        h = self.main(x)              # (B, dim, h', w')
        h = self.activation(h)
        h = self.pool(h)              # (B, dim, 1,1)
        h = h.view(h.size(0), -1)     # (B, dim)
        out = self.fc(h)              # (B, num_domains)
        # 각 배치별 y 인덱스에 해당하는 점수만 리턴해서 LSGAN MSELoss 용도로 사용
        idx = torch.arange(x.size(0), device=x.device)
        return out[idx, y].unsqueeze(1)  # (B, 1)
