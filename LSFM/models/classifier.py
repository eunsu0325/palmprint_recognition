import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Multi-layer classifier head.
    Returns a list of features (outputs of each Linear) and final logits.
    """
    def __init__(self, in_dim, hidden_dims, num_classes):
        super().__init__()
        dims = [in_dim] + hidden_dims + [num_classes]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        out = x
        for m in self.model:
            out = m(out)
            if isinstance(m, nn.Linear):
                features.append(out)
        logits = features[-1]
        return features, logits