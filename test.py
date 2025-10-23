import os
import sys
import json
import torch
import logging
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from general import utils, viz
from general.model import CosineMLP
from general.arg import parse_test_args
from general.dataset import EmbeddingDataset
from general.eval import evaluate, mc_dropout_eval, single_sample


def reconstruct_model(
        run_dir: Path,
        emb_dim: int,
        num_classes: int,
        device: torch.device
) -> nn.Module:
    """
    Rebuild CosineMLP from checkpoint shapes.
    - hidden_dim inferred from first MLP layer or fc weight.
    - initial scale pulled from checkpoint if present.
    """
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    state = ckpt["model_state"]

    hidden_dim = None
    for k in ["mlp.0.weight", "layers.0.weight", "encoder.0.weight"]:
        if k in state:
            hidden_dim = state[k].shape[0]  # (hidden_dim, emb_dim)
            break
    if hidden_dim is None:
        if "fc.weight" in state:
            hidden_dim = state["fc.weight"].shape[1]  # (num_classes, hidden_dim)
        else:
            raise RuntimeError("Cannot infer hidden_dim from checkpoint.")

    init_scale = 20.0
    if "scale" in state and state["scale"].ndim == 0:
        init_scale = float(state["scale"].item())

    model = CosineMLP(
        in_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.5,         # dropout prob doesn't affect state_dict loading
        scale=init_scale,
    ).to(device)

    model.load_state_dict(state, strict=True)
    return model


def main():
    utils.set_logging()
    args = parse_test_args()
    utils.set_seed(args.seed)

    # Get model directory
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        logging.error(f"Run dir not found: {run_dir}")
        sys.exit(1)

    # Load metadata
    with open(run_dir / "label_map.json", "r") as f:
        label_map = json.load(f)
    with open(run_dir / "splits.json", "r") as f:
        splits = json.load(f)

    # Get list of test file paths
    test_files = [Path(p) for p in splits["test"]]
    if not test_files:
        logging.error("Test split is empty.")
        sys.exit(1)

    embe_root = Path(os.path.commonpath([str(p.parent) for p in test_files]))
    logging.info(f"Embeddings root: {embe_root}")

    embe_dim = utils.infer_embedding_dim(test_files[0])
    logging.info(f"Embeddings dimension: {embe_dim}")

    num_classes = len(label_map)
    logging.info(f"Classes: {num_classes}")

    test_ds = EmbeddingDataset(test_files, embe_root, label_map)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    model = reconstruct_model(run_dir, embe_dim, num_classes, args.device)

    T = int(args.mc_passes)

    # Deterministic eval (dropout OFF)
    test_loss, test_acc = evaluate(model, test_loader, args.device)
    print(f"[Deterministic] test_loss={test_loss:.4f}, acc={test_acc:.4f}")

    # MC-Dropout eval (dropout ON, T passes)
    acc_mean, ent_mean, vr_mean, entropies, correct, labels = mc_dropout_eval(
        model, test_loader, args.device, T=T
    )
    entropies = entropies.numpy()
    correct = correct.numpy()
    labels = labels.numpy()
    print(
        f"[MC-Dropout T={T}]"
        f", mean accuracy={acc_mean:.4f}"
        f", mean entropy={ent_mean:.4f}"
        f", mean variance ration={vr_mean:.4f}"
    )

    # Visualizations
    probs_T, y = single_sample(model, test_loader, args.device, T)
    fig = viz.fig_sample(probs_T)
    fig.savefig(run_dir / "sample_mc_boxplot.jpg", dpi=150)

    fig = viz.fig_uncertainty_vs_correctness(entropies, correct)
    fig.savefig(run_dir / "uncertainty_vs_correctness.jpg", dpi=150)

    fig = viz.fig_per_speaker_ent_acc(labels, entropies, correct)
    fig.savefig(run_dir / "per_speaker_ent_acc.jpg", dpi=150)

    fig = viz.fig_pca_embeddings(
        test_files,
        entropies=entropies,
        correct_mask=correct,
        color="entropy"
    )
    fig.savefig(run_dir / "pca_by_entropy.jpg", dpi=150)

    fig = viz.fig_pca_embeddings(
        test_files,
        entropies=entropies,
        correct_mask=correct,
        color="correct"
    )
    fig.savefig(run_dir / "pca_by_correct.jpg", dpi=150)


if __name__ == "__main__":
    main()
