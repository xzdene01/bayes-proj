import torch
import torch.nn as nn

from typing import Tuple, List
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())
        total += yb.size(0)
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def mc_dropout_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int = 30,
) -> Tuple[float, float, float]:
    """
    Returns:
        acc_mean       : accuracy using mean predictive probabilities
        mean_entropy   : average predictive entropy over samples
        mean_var_ratio : average variation ratio (1 - max prob)
    """
    model.train()  # enable dropout
    total_correct, total = 0, 0
    ent_sum, vr_sum = 0.0, 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        probs_T: List[torch.Tensor] = []
        for _ in range(T):
            logits = model(xb)
            probs_T.append(F.softmax(logits, dim=1))

        probs_mean = torch.stack(probs_T, dim=0).mean(dim=0)  # (B, C)
        preds = probs_mean.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())
        total += yb.size(0)

        eps = 1e-12
        entropy = -(probs_mean * (probs_mean + eps).log()).sum(dim=1)  # (B,)
        var_ratio = 1.0 - probs_mean.max(dim=1).values                  # (B,)

        ent_sum += float(entropy.sum().item())
        vr_sum += float(var_ratio.sum().item())

    acc_mean = total_correct / max(1, total)
    mean_entropy = ent_sum / max(1, total)
    mean_var_ratio = vr_sum / max(1, total)
    return acc_mean, mean_entropy, mean_var_ratio
