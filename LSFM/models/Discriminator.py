import math
import torch
import torch.nn as nn
from core.model_blocks import ResBlk  # StarGAN v2 ResBlk

class Discriminator(nn.Module):
    """
    StarGAN v2 원본 구조를 따르는 다중 도메인 패치GAN 판별기 구현
    입력: (B, C, H, W) palmprint image
    출력: (B,) 해당 도메인 Real/Fake logit
    """
    def __init__(self,
                 img_size: int = 256,
                 in_channels: int = 1,
                 num_domains: int = 4,
                 max_conv_dim: int = 512):
        super().__init__()
        # 초기 채널 수 설정
        dim_in = 2**14 // img_size
        blocks = []
        # from_rgb conv
        blocks.append(nn.Conv2d(in_channels, dim_in, 3, 1, 1))
        # 다운샘플링 ResBlk 반복
        repeat_num = int(math.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            dim_in = dim_out
        # 최종 레이어: LeakyReLU → 4×4 Conv → LeakyReLU → 도메인별 1×1 Conv
        blocks += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, num_domains, 1, 1, 0)
        ]
        self.main = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, domain_labels: torch.LongTensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, num_domains, H', W')
        out = self.main(x)
        # Flatten spatial dims: (B, num_domains, H'*W') -> (B, num_domains, -1)
        out = out.view(out.size(0), out.size(1), -1)
        # 평균 풀링 or sum: 하나의 로그잇으로 압축
        out = out.mean(dim=2)  # (B, num_domains)
        # 도메인별 로짓 선택
        idx = torch.arange(x.size(0), device=x.device)
        logits = out[idx, domain_labels]  # (B,)
        return logits

if __name__ == '__main__':
    D = Discriminator()
    x = torch.randn(2,1,256,256)
    labels = torch.tensor([0,3])
    out = D(x, labels)
    print(out.shape)  # [2]
