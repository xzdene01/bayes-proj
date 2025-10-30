import argparse


def parse_train_args():
    ap = argparse.ArgumentParser(description="Train MLP + Cosine classifier")

    ap.add_argument(
        "--embes",
        type=str, required=True,
        help="Root folder with .npy embeddings"
    )
    ap.add_argument(
        "--out",
        type=str, default="./runs/tmp",
        help="Where to save model and all files created during training"
    )
    ap.add_argument(
        "--device",
        type=str, default="cpu",
        choices=["cpu", "cuda", "mps"]
    )

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=40.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def parse_test_args():
    ap = argparse.ArgumentParser(description="Evaluate MLP with MC Dropout")

    ap.add_argument(
        "--run",
        type=str, required=True,
        help="Directory with model.pt, label_map.json, and splits.json"
    )
    ap.add_argument(
        "--mc_passes",
        type=int, default=100,
        help="Number of Monte Carlo dropout passes"
    )
    ap.add_argument(
        "--device",
        type=str, default="cpu",
        choices=["cpu", "cuda", "mps"]
    )

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()
