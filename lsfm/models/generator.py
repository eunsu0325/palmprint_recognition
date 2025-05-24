"""
LightStarGAN Generator
- 논문 『LSFM: Light-Style & Feature Matching …』 구현
- Encoder 6층 ↔ Decoder 6층 대칭 구조
"""
import torch
import torch.nn as nn

# -------------------------------------------------
# utility : channel-shuffle
# -------------------------------------------------
def channel_shuffle(x, groups: int = 2):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w)        # (B, g, c/g, H, W)
    x = torch.transpose(x, 1, 2).contiguous()       # (B, c/g, g, H, W)
    return x.view(b, c, h, w)                       # (B, C, H, W)


# -------------------------------------------------
# Relaxed ShuffleNet-v2 block
# -------------------------------------------------
class ShuffleV2Block(nn.Module):
    """
    stride==1 & inp==oup  : 오리지널 split-cat
    stride==1 & inp!=oup  : 1×1 conv → split-cat
    stride==2             : downsample variant
    """
    def __init__(self, inp: int, oup: int, stride: int = 1):
        super().__init__()
        assert stride in (1, 2), "stride must be 1 or 2"
        self.stride = stride
        mid = oup // 2

        # branch-1 --------------------------------------------------
        if stride == 1 and inp == oup:
            self.branch1 = nn.Identity()            # 그대로 split 사용
        else:                                       # stride==2 or  channel change
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True)
            )

        # branch-2 --------------------------------------------------
        in_b2 = inp if (stride > 1 or inp != oup) else mid
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_b2, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1 and isinstance(self.branch1, nn.Identity):
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


# -------------------------------------------------
# AdaIN
# -------------------------------------------------
class AdaIN(nn.Module):
    def __init__(self, style_dim: int, num_feat: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_feat, affine=False)
        self.fc   = nn.Linear(style_dim, num_feat * 2)

    def forward(self, x, s):
        h = self.fc(s).view(s.size(0), -1, 1, 1)
        gamma, beta = h.chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


# -------------------------------------------------
# Light encoder / decoder layers
# -------------------------------------------------
class LightEncoderLayer(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.block = ShuffleV2Block(inp, oup, stride)

    def forward(self, x): return self.block(x)


class LightDecoderLayer(nn.Module):
    def __init__(self, inp, oup, style_dim, upsample: bool = False):
        super().__init__()
        self.adain    = AdaIN(style_dim, inp)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else None
        self.block    = ShuffleV2Block(inp, oup, stride=1)

    def forward(self, x, s):
        x = self.adain(x, s)
        if self.upsample is not None:
            x = self.upsample(x)
        return self.block(x)


# -------------------------------------------------
# Generator
# -------------------------------------------------
class Generator(nn.Module):
    def __init__(self, style_dim: int = 64, in_channels: int = 1):
        super().__init__()

        # Encoder ---------------------------------------------------
        self.enc1 = LightEncoderLayer(in_channels,  64, stride=2)
        self.enc2 = LightEncoderLayer(64,  128, stride=2)
        self.enc3 = LightEncoderLayer(128, 128, stride=2)
        self.enc4 = LightEncoderLayer(128, 256, stride=2)
        self.enc5 = LightEncoderLayer(256, 256, stride=1)
        self.enc6 = LightEncoderLayer(256, 256, stride=1)

        # Decoder ---------------------------------------------------
        self.dec6 = LightDecoderLayer(256, 256, style_dim, upsample=False)
        self.dec5 = LightDecoderLayer(256, 256, style_dim, upsample=False)
        self.dec4 = LightDecoderLayer(256, 128, style_dim, upsample=True)
        self.dec3 = LightDecoderLayer(128, 128, style_dim, upsample=True)
        self.dec2 = LightDecoderLayer(128,  64, style_dim, upsample=True)
        self.dec1 = LightDecoderLayer( 64,  64, style_dim, upsample=True)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, in_channels, 1, 1, 0),
            nn.Tanh()
        )

    # -------------------------------------------------------------
    def forward(self, x, s):
        # Encode
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        e4 = self.enc4(e3); e5 = self.enc5(e4); e6 = self.enc6(e5)

        # Decode (skip-connection)
        d6 = self.dec6(e6, s)
        d5 = self.dec5(d6 + e5, s)
        d4 = self.dec4(d5 + e4, s)
        d3 = self.dec3(d4 + e3, s)
        d2 = self.dec2(d3 + e2, s)
        d1 = self.dec1(d2 + e1, s)

        return self.to_rgb(d1)


# -----------------------------------------------------------------
# quick unit-test
# -----------------------------------------------------------------
if __name__ == "__main__":
    G = Generator()
    x = torch.randn(2, 1, 128, 128)
    s = torch.randn(2, 64)
    y = G(x, s)
    print("output :", y.shape)   # -> torch.Size([2, 1, 128, 128])
