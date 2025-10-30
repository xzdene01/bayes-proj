import sys
import json
import torch
import logging
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from general.eval import evaluate
from general.model import CosineMLP
from general.arg import parse_train_args
from general.dataset import EmbeddingDataset
from general.utils import (
    set_seed,
    build_label_map,
    infer_embedding_dim,
    set_logging
)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # Accumulate loss and acc for each epoch
            total_loss += float(loss.item()) * yb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())
            total += yb.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = total_correct / max(1, total)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device)

        logging.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"s={model.scale.item():.2f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "scale": float(model.scale.item())
        })

        # Save best model by val acc
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta": {"best_val_acc": best_val_acc},
                },
                outdir / "model.pt",
            )

    # Save full training history
    with open(outdir / "training_summary.json", "w") as f:
        json.dump(history, f, indent=2)


def main():
    set_logging()
    args = parse_train_args()
    set_seed(args.seed)

    logging.debug(args)

    embeddings_root = Path(args.embes).resolve()
    outdir = Path(args.out).resolve()
    device = torch.device(args.device)

    all_files = [p for p in embeddings_root.rglob("*.npy") if p.is_file()]
    if not all_files:
        logging.error(f"No .npy files found in {embeddings_root}")
        sys.exit(1)
    logging.info(f"Found {len(all_files)} files in {embeddings_root}")

    label_map = build_label_map(all_files, embeddings_root)
    num_classes = len(label_map)
    logging.info(f"Found {num_classes} speakers")

    # Save label map for later
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    emb_dim = infer_embedding_dim(all_files[0])
    logging.info(f"Embedding dimension: {emb_dim}")

    # Extract speaker labels
    labels = [
        label_map[p.relative_to(embeddings_root).parts[0]]
        for p in all_files
    ]

    # First split: 70/30 (train/val+test)
    train_files, temp_files, y_train, y_temp = train_test_split(
        all_files, labels, stratify=labels, test_size=0.3,
        random_state=args.seed
    )

    # Second split: 15/15 (val/test)
    val_files, test_files, y_val, y_test = train_test_split(
        temp_files, y_temp, stratify=y_temp, test_size=0.5,
        random_state=args.seed
    )

    logging.info(
        f"Splits: train: {len(train_files)}, val: {len(val_files)}, "
        f"test: {len(test_files)}"
    )

    # Save the splits
    splits = {
        "train": [str(p) for p in train_files],
        "val": [str(p) for p in val_files],
        "test": [str(p) for p in test_files],
    }

    with open(outdir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    train_ds = EmbeddingDataset(train_files, embeddings_root, label_map)
    val_ds = EmbeddingDataset(val_files, embeddings_root, label_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    model = CosineMLP(
        in_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        scale=args.scale,
    ).to(device)

    train(
        model, train_loader, val_loader,
        device, args.epochs, args.lr, outdir
    )
    logging.info(f"Training done. Model saved to: {outdir}")


if __name__ == "__main__":
    main()
