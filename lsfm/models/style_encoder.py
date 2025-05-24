# ─────────────────────────────────────────────────────────────
#  lsfm/models/style_encoder.py   (논문 LSFM/StarGAN-v2 호환)
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# 프로젝트에서 이미 쓰고 있는 ResBlk 가져오기
try:
    from core.model import ResBlk          # 기존 구현
except ModuleNotFoundError:
    # 최소 대체 구현 (stride=2 downsample 시 사용)
    class ResBlk(nn.Module):
        def __init__(self, dim_in, dim_out, downsample=False):
            super().__init__()
            self.down = downsample
            self.learned_sc = (dim_in != dim_out)
            self.conv1 = nn.Conv2d(dim_in,  dim_out, 3, 1, 1)
            self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
            self.act   = nn.LeakyReLU(0.2, inplace=True)
            if downsample:
                self.avgpool = nn.AvgPool2d(2)
            if self.learned_sc:
                self.conv_sc = nn.Conv2d(dim_in, dim_out, 1, 1, 0)

        def shortcut(self, x):
            h = x
            if self.down:    h = self.avgpool(h)
            if self.learned_sc: h = self.conv_sc(h)
            return h

        def residual(self, x):
            h = self.conv1(x)
            h = self.act(h)
            h = self.conv2(h)
            if self.down: h = self.avgpool(h)
            return h

        def forward(self, x):
            return self.residual(x) + self.shortcut(x)


class StyleEncoder(nn.Module):
    """
    LSFM / StarGAN v2 스타일 인코더
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

        # 1. 첫 conv 채널 수 (StarGAN v2 룰)
        dim_in = 2 ** 14 // img_size      # 128→64, 256→32 …
        layers = [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        # 2. 다운샘플 잔차블록 N 회 (log2(img)-2)
        repeat_num = int(torch.log2(torch.tensor(img_size)).item()) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            layers.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out                            # 다음 입력 채널

        # 3. 4×4 → 1×1
        layers += [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_in, 4, 1, 0),         # dim_in 사용!
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.shared = nn.Sequential(*layers)

        # 4. 도메인별 완전 연결
        self.unshared = nn.ModuleList(
            [nn.Linear(dim_in, style_dim) for _ in range(num_domains)]
        )
        self.style_dim   = style_dim
        self.num_domains = num_domains

    # ------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.LongTensor):
        """
        x : (B, C, H, W)  gray palm image
        y : (B,)          domain index tensor (int64 권장)
        """
        if x.dim() != 4:
            raise ValueError("input must be 4-D (B,C,H,W)")
        y = y.to(torch.long)

        h = self.shared(x)            # (B, C, 1, 1)
        h = h.view(h.size(0), -1)     # (B, C)
        style_all = torch.stack([fc(h) for fc in self.unshared], dim=1)
        idx = torch.arange(x.size(0), device=x.device)
        return style_all[idx, y]      # (B, style_dim)


# --------------------------- quick self-test ---------------------------
if __name__ == "__main__":
    enc = StyleEncoder(img_size=128, style_dim=64, num_domains=4, in_channels=1)
    x   = torch.randn(4, 1, 128, 128)
    dom = torch.tensor([0, 2, 1, 3])
    s   = enc(x, dom)
    print("style code shape :", s.shape)   # torch.Size([4, 64])
