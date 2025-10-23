import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


# One-sample visualization (MC dropout per-class distribution)
def fig_sample(
    probs_T: torch.Tensor,  # (T, C) softmax probs for a single sample
    class_names: Optional[List[str]] = None,
    top_k: int = 10,
    title: str = "MC-Dropout: per-class probability spread (one sample)"
):
    assert probs_T.ndim == 2, "probs_T must be (T, C)"
    T, C = probs_T.shape
    p = probs_T.detach().cpu().numpy()
    mean_p = p.mean(axis=0)
    top_idx = np.argsort(mean_p)[::-1][:max(1, min(top_k, C))]

    data = [p[:, i] for i in top_idx]
    labels = [class_names[i] if class_names else f"c{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot(data, showfliers=False)

    ax.set_title(title)
    ax.set_xlabel("Top-K classes")
    ax.set_ylabel("Probability")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    return fig


# Uncertainty vs correctness (boxplot over the whole test set)
def fig_uncertainty_vs_correctness(
    entropies: np.ndarray,       # shape (N,)
    correct_mask: np.ndarray,    # shape (N,) bool
    title: str = "Uncertainty vs Correctness (entropy)"
):
    entropies = np.asarray(entropies, dtype=float)
    correct_mask = np.asarray(correct_mask, dtype=bool)

    corr = entropies[correct_mask]
    inc = entropies[~correct_mask]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([corr, inc], showfliers=False, labels=["Correct", "Incorrect"])

    ax.set_title(title)
    ax.set_ylabel("Predictive entropy")
    return fig


# Per-speaker mean entropy + accuracy
def fig_per_speaker_ent_acc(
    speaker_ids: np.ndarray,        # (N,)
    entropies: np.ndarray,          # (N,)
    correct_mask: np.ndarray,       # (N,)
    k: int = 100,
    title: str = "Per-speaker uncertainty (mean entropy) and accuracy",
    seed: int = 42
):
    speaker_ids = np.asarray(speaker_ids, dtype=int)
    entropies = np.asarray(entropies, dtype=float)
    correct_mask = np.asarray(correct_mask, dtype=bool)

    # Aggregate per speaker
    sums_ent = defaultdict(float)
    counts = defaultdict(int)
    sums_acc = defaultdict(int)
    for spk, e, ok in zip(speaker_ids, entropies, correct_mask):
        sums_ent[spk] += float(e)
        counts[spk] += 1
        sums_acc[spk] += int(ok)

    spk_list = list(counts.keys())
    mean_ent = np.array([sums_ent[s]/counts[s] for s in spk_list])
    acc_spk = np.array([sums_acc[s]/counts[s] for s in spk_list])

    # Select K random speakers
    np.random.seed(seed)
    idxs = np.random.choice(
        len(spk_list), size=min(k, len(spk_list)),
        replace=False
    )
    idxs = sorted(idxs, key=lambda i: mean_ent[i])

    spk_sel = [spk_list[i] for i in idxs]
    ent_sel = mean_ent[idxs]
    acc_sel = acc_spk[idxs]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(len(spk_sel)), ent_sel, color="tab:blue", alpha=0.5)

    ax1.set_title(title)
    ax1.set_xlabel("Speakers")
    ax1.set_ylabel("Mean entropy")
    # ax1.set_xticks(range(len(spk_sel)))
    # ax1.set_xticklabels(spk_sel, rotation=45, ha="right")  # too many spks

    # Hide x ticks + labels, not important
    ax1.tick_params(axis='x', bottom=False, labelbottom=False)

    # Overlay accuracy
    ax2 = ax1.twinx()
    ax2.plot(range(len(spk_sel)), acc_sel, "k-", linewidth=0.5)
    ax2.set_ylabel("Accuracy")

    fig.tight_layout()
    return fig


def fig_pca_embeddings(
    test_files,
    entropies=None,
    correct_mask=None,
    color="entropy",  # entropy/correct/none
    random_state=42,
    title="PCA of Test Embeddings"
):
    # Load embeddings
    X_list = []
    tf = [Path(p) for p in test_files]
    for p in tf:
        e = np.load(p)
        e = np.asarray(e).reshape(-1)
        X_list.append(e.astype(np.float32))

    X = np.stack(X_list, axis=0)  # (N, D)

    # Optional coloring
    ent = None
    if entropies is not None:
        ent = np.asarray(entropies, dtype=np.float32)

    corr = None
    if correct_mask is not None:
        corr = np.asarray(correct_mask, dtype=bool)

    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)  # (k, 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    if color == "entropy" and ent is not None:
        sc = ax.scatter(
            Z[:, 0], Z[:, 1], c=ent, s=3, cmap="viridis",
            alpha=0.8, edgecolors="none"
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Predictive Entropy")
        ax.set_title(title + " (colored by entropy)")

    elif color == "correct" and corr is not None:
        c = np.where(corr, "tab:blue", "tab:red")
        sc = ax.scatter(
            Z[:, 0], Z[:, 1], c=c, s=3,
            alpha=0.8, edgecolors="none"
        )
        # 'Fake' plot, just to show legend correctly for the scatter
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", label="Correct",
                markerfacecolor="tab:blue", markersize=6
            ),
            Line2D(
                [0], [0], marker="o", color="w", label="Incorrect",
                markerfacecolor="tab:red", markersize=6
            )
        ]
        ax.legend(handles=handles)
        ax.set_title(title + " (correct vs incorrect)")

    else:
        ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.8, edgecolors="none")
        ax.set_title(title)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.tight_layout()
    return fig
