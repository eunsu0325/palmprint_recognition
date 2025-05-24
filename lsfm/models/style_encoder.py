# ──────────────────────────────────────────────────────────────
#  lsfm/models/style_encoder.py  (전체 코드 교체본)
# ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import ResBlk          # StarGAN v2와 동일한 잔차 블록

class StyleEncoder(nn.Module):
    """
    StarGAN v2-형 LightStarGAN Style Encoder

    Args:
        img_size     (int): 입력 해상도. 128·256 등 2의 거듭제곱.
        style_dim    (int): 스타일 벡터 차원.
        num_domains  (int): 도메인(뷰) 개수.
        max_conv_dim (int): conv 채널 상한선.
        in_channels  (int): 입력 영상 채널 수. (기본 1: palmprint gray)
    """
    def __init__(
        self,
        img_size    : int  = 128,
        style_dim   : int  = 64,
        num_domains : int  = 4,
        max_conv_dim: int  = 512,
        in_channels : int  = 1
    ):
        super().__init__()

        # StarGAN v2 관습: 첫 conv 채널 = 2^14 / img_size
        dim_in = 2 ** 14 // img_size
        blocks = []

        # 1) from-rgb
        blocks.append(nn.Conv2d(in_channels, dim_in, 3, 1, 1))

        # 2) 다운샘플 residual 블록
        repeat_num = int(torch.log2(torch.tensor(img_size)).item()) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        # 3) 마지막 4×4 → 1×1 map
        blocks += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2)
        ]
        self.shared = nn.Sequential(*blocks)

        # 4) 도메인-별 fully-connected (unshared)
        self.unshared = nn.ModuleList([
            nn.Linear(dim_out, style_dim) for _ in range(num_domains)
        ])
        self.style_dim   = style_dim
        self.num_domains = num_domains

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.LongTensor):
        """
        x : [B, C, H, W]  - palmprint image
        y : [B]           - domain label (0 ~ num_domains-1)
        return : [B, style_dim]
        """
        h = self.shared(x)           # [B, dim_out, 1, 1]
        h = h.view(h.size(0), -1)    # [B, dim_out]

        # 모든 도메인 fc 통과 후 스택 → [B, D, style_dim]
        s_all = torch.stack([fc(h) for fc in self.unshared], dim=1)

        # y에 해당하는 style-code 선택
        idx = torch.arange(x.size(0), device=x.device)
        s   = s_all[idx, y]          # [B, style_dim]
        return s


# ──────────────────────────────────────────────────────────────
#  간단 self-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    enc = StyleEncoder(img_size=128,
                       style_dim=64,
                       num_domains=4,
                       in_channels=1)
    x = torch.randn(4, 1, 128, 128)
    dom = torch.tensor([0, 2, 1, 3])
    s = enc(x, dom)
    print("style code shape:", s.shape)   # → torch.Size([4, 64])
