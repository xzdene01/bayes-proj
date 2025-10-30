import torch
import random
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
    model.train()  # enable dropout
    total_correct, total = 0, 0
    ent_sum, vr_sum = 0.0, 0.0
    all_entropies, all_correct, all_labels = [], [], []

    # Process dataset in batches
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        # Preds per sample for T passes
        probs_T: List[torch.Tensor] = []
        for _ in range(T):
            logits = model(xb)  # (B, C)
            probs_T.append(F.softmax(logits, dim=1))

        # Mean distribution per sample
        probs_T = torch.stack(probs_T, dim=0)  # (T, B, C)
        probs_mean = probs_T.mean(dim=0)  # (B, C)

        # Class per sample
        preds = probs_mean.argmax(dim=1)  # (B,)

        total_correct += int((preds == yb).sum().item())
        total += yb.size(0)

        eps = 1e-12
        entropy = -(probs_mean * (probs_mean + eps).log()).sum(dim=1)   # (B,)
        var_ratio = 1.0 - probs_mean.max(dim=1).values  # (B,)
        ent_sum += float(entropy.sum().item())
        vr_sum += float(var_ratio.sum().item())

        all_entropies.append(entropy.cpu())
        all_correct.append((preds == yb).cpu())
        all_labels.append(yb.cpu())

    acc_mean = total_correct / max(1, total)
    mean_entropy = ent_sum / max(1, total)
    mean_var_ratio = vr_sum / max(1, total)

    entropies = torch.cat(all_entropies)
    correct = torch.cat(all_correct)
    labels = torch.cat(all_labels)
    return acc_mean, mean_entropy, mean_var_ratio, entropies, correct, labels


@torch.no_grad()
def single_sample(model, loader, device, T=30, seed=42):
    model.train()
    random.seed(seed)

    loader_iter = iter(loader)
    batches = list(loader_iter)

    batch_idx = random.randint(0, len(batches) - 1)
    xb, yb = batches[batch_idx]

    item_idx = random.randint(0, xb.size(0) - 1)
    xb = xb[item_idx:item_idx + 1].to(device)  # (1, D)
    yb = yb[item_idx].item()  # scalar

    probs_T = []
    for _ in range(T):
        logits = model(xb)  # (1, C)
        probs = F.softmax(logits, dim=1)  # (1, C)
        probs_T.append(probs.squeeze(0))  # (C,)

    probs_T = torch.stack(probs_T, dim=0)  # (T, C)
    return probs_T.cpu(), yb
