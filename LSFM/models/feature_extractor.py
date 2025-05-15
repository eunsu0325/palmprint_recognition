import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    """
    Backbone feature extractor for LSFM using VGG16 pretrained conv layers.
    Outputs a flattened feature vector for classification and MK-MMD alignment.
    """
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        # Load pretrained VGG16-BN model
        vgg = models.vgg16_bn(pretrained=pretrained)
        # If input is not RGB, adjust first conv weights
        if in_channels != 3:
            old_conv = vgg.features[0]
            new_conv = nn.Conv2d(in_channels,
                                 old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))
            with torch.no_grad():
                # Initialize new conv by averaging pretrained weights across channels
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # (out_c,1,k,k)
                new_conv.weight.copy_(mean_w.repeat(1, in_channels, 1, 1) / in_channels)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            vgg.features[0] = new_conv
        # Use VGG feature layers and avgpool
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        # Determine output dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            feat = self.features(dummy)
            pooled = self.avgpool(feat)
        self.out_dim = pooled.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
        Returns:
            Tensor: Flattened feature vector of shape (B, out_dim)
        """
        h = self.features(x)
        h = self.avgpool(h)
        return h.view(h.size(0), -1)

if __name__ == '__main__':
    F = FeatureExtractor()
    x = torch.randn(2, 1, 128, 128)
    out = F(x)
    print(out.shape, F.out_dim)  # expected (2, out_dim)
