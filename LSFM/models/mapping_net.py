import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    """
    StarGAN v2 기반 LightStarGAN 매핑 네트워크 구현

    입력:
      z              (B, latent_dim)       - 랜덤 노이즈 벡터
      domain_labels  (B,)                   - 정수 도메인 레이블

    출력:
      style_codes    (B, style_dim)
    """
    def __init__(self,
                 latent_dim: int = 16,
                 style_dim: int = 64,
                 num_domains: int = 4):
        super().__init__()
        # 1) Shared MLP: latent_dim → 512 → 512 → 512 → 512
        layers = [nn.Linear(latent_dim, 512), nn.ReLU(inplace=True)]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU(inplace=True)]
        self.shared = nn.Sequential(*layers)

        # 2) Unshared MLPs: 도메인별로 512 → 512 → 512 → 512 → style_dim
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 512), nn.ReLU(inplace=True),  # 추가된 레이어
                nn.Linear(512, style_dim)
            ))

    def forward(self, z: torch.Tensor, domain_labels: torch.LongTensor):
        """
        z:               (B, latent_dim)
        domain_labels:   (B,)
        → style_codes:   (B, style_dim)
        """
        h = self.shared(z)                         # (B, 512)
        # 도메인별 unshared MLP 통과
        outs = [m(h) for m in self.unshared]       # list of (B, style_dim)
        stacked = torch.stack(outs, dim=1)         # (B, num_domains, style_dim)
        # 배치별 도메인 인덱싱
        idx = torch.arange(z.size(0), device=z.device)
        style_codes = stacked[idx, domain_labels]  # (B, style_dim)
        return style_codes

if __name__ == '__main__':
    M = MappingNetwork()
    z = torch.randn(2, 16)
    labels = torch.tensor([0, 3])
    s = M(z, labels)
    print(s.shape)  # [2, 64]
