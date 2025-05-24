import torch
import torch.nn as nn

# ShuffleNetV2 unit (single stage) with channel split & shuffle
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose group and channel dimensions
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.stride = stride
        mid_channels = oup // 2
        if stride == 1:
            assert inp == oup, "Input and output channels must match when stride=1"
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        # branch2 always exists
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride>1 else mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)

# Adaptive Instance Norm with prior InstanceNorm
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # x: feature map, s: style vector
        x_norm = self.norm(x)
        h = self.fc(s).unsqueeze(2).unsqueeze(3)
        gamma, beta = h.chunk(2, dim=1)
        return (1 + gamma) * x_norm + beta

# LightEncoderLayer: efficient downsampling block
class LightEncoderLayer(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.block = ShuffleV2Block(inp, oup, stride)

    def forward(self, x):
        return self.block(x)

# LightDecoderLayer: AdaIN style injection + upsampling + efficient conv
class LightDecoderLayer(nn.Module):
    def __init__(self, inp, oup, style_dim, upsample=False):
        super().__init__()
        self.adain = AdaIN(style_dim, inp)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else None
        self.conv = ShuffleV2Block(inp, oup, stride=1)

    def forward(self, x, s):
        x = self.adain(x, s)
        if self.upsample:
            x = self.upsample(x)
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, style_dim=64, in_channels=1):
        super().__init__()
        # Encoder: 6 layers, first 4 with stride=2, last 2 with stride=1
        self.enc1 = LightEncoderLayer(in_channels, 64, stride=2)
        self.enc2 = LightEncoderLayer(64, 128, stride=2)
        self.enc3 = LightEncoderLayer(128, 128, stride=2)
        self.enc4 = LightEncoderLayer(128, 256, stride=2)
        self.enc5 = LightEncoderLayer(256, 256, stride=1)
        self.enc6 = LightEncoderLayer(256, 256, stride=1)

        # Decoder: symmetric 6 layers, first 4 upsample
        self.dec6 = LightDecoderLayer(256, 256, style_dim, upsample=False)
        self.dec5 = LightDecoderLayer(256, 256, style_dim, upsample=False)
        self.dec4 = LightDecoderLayer(256, 128, style_dim, upsample=True)
        self.dec3 = LightDecoderLayer(128, 128, style_dim, upsample=True)
        self.dec2 = LightDecoderLayer(128, 64, style_dim, upsample=True)
        self.dec1 = LightDecoderLayer(64, 64, style_dim, upsample=True)

        # Final conversion back to image
        self.final = nn.Sequential(
            nn.Conv2d(64, in_channels, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x, s):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        # Decoding with skip-connections
        d6 = self.dec6(e6, s)
        d5 = self.dec5(d6 + e5, s)
        d4 = self.dec4(d5 + e4, s)
        d3 = self.dec3(d4 + e3, s)
        d2 = self.dec2(d3 + e2, s)
        d1 = self.dec1(d2 + e1, s)

        return self.final(d1)

if __name__ == '__main__':
    G = Generator()
    x = torch.randn(2, 1, 128, 128)
    s = torch.randn(2, 64)
    out = G(x, s)
    print(out.shape)  # expected [2, 1, 128, 128]
