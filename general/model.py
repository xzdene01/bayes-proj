import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineMLP(nn.Module):
    """
    MLP with dropout, followed by cosine-normalized classifier.
    logits = scale * normalize(h) @ normalize(W).T
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale),
                                               dtype=torch.float32))

        # Kaiming init for backbone linear, and unit-norm-ish init for fc
        nn.init.kaiming_normal_(self.backbone[1].weight, nonlinearity='relu')
        nn.init.zeros_(self.backbone[1].bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        h = self.backbone(x)  # (B, H)

        # Normalize features and class weights
        h = F.normalize(h, dim=1)
        W = F.normalize(self.fc.weight, dim=1)  # (C, H) - will be transposed
        logits = self.scale * torch.matmul(h, W.t())  # (B, C)
        return logits
