import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import ResBlk

class StyleEncoder(nn.Module):
    """
    StarGAN v2 원본 구조를 그대로 따라간 LSFM 스타일 인코더 구현.

    Args:
        img_size    (int): 입력 이미지 해상도 (예: 256).
        style_dim   (int): 출력 스타일 코드 차원 (예: 64).
        num_domains (int): 도메인 수 (예: 4).
        max_conv_dim(int): 최대 채널 수 (예: 512).
    """
    def __init__(self,
                 img_size: int = 256,
                 style_dim: int = 64,
                 num_domains: int = 4,
                 max_conv_dim: int = 512):
        super().__init__()
        # 최초 채널 크기: 2^14 / img_size (StarGAN v2 관행)
        dim_in = 2**14 // img_size
        blocks = []
        # 1) from_rgb
        blocks.append(nn.Conv2d(1, dim_in, 3, 1, 1))
        # 2) down-sampling ResBlks
        repeat_num = int(torch.log2(torch.tensor(img_size)).item()) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        # 3) 최종 맵 → 1×1 피처
        blocks += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2)
        ]
        self.shared = nn.Sequential(*blocks)

        # 4) Unshared FC: 각 도메인별로 style_dim 생성
        self.unshared = nn.ModuleList([
            nn.Linear(dim_out, style_dim)
            for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, y: torch.LongTensor):
        """
        x: (B, 1, H, W), y: (B,)
        returns: style_codes (B, style_dim)
        """
        h = self.shared(x)                 # (B, dim_out, 1, 1)
        h = h.view(h.size(0), -1)          # (B, dim_out)
        outs = [fc(h) for fc in self.unshared]       # list of (B, style_dim)
        out = torch.stack(outs, dim=1)               # (B, num_domains, style_dim)
        idx = torch.arange(x.size(0), device=x.device)
        s = out[idx, y]                              # (B, style_dim)
        return s

if __name__ == '__main__':
    E = StyleEncoder()
    x = torch.randn(2,1,256,256)
    labels = torch.tensor([0,2])
    s = E(x, labels)
    print(s.shape)  # [2, 64]
