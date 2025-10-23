import sys
import torch
import random
import logging
import numpy as np

from pathlib import Path
from typing import Dict, List


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_map(files: List[Path], embeddings_root: Path) \
        -> Dict[str, int]:
    """
    Build a mapping from speaker folder names to integer labels.

    Each top-level subfolder under 'embeddings_root' represents a unique
    speaker. All files inside that subfolder are assigned to the same speaker.
    """
    speakers = set()
    for p in files:
        rel = p.relative_to(embeddings_root)
        spk = rel.parts[0]
        speakers.add(spk)
    speakers = sorted(speakers)
    return {spk: i for i, spk in enumerate(speakers)}


def infer_embedding_dim(sample_file: Path) -> int:
    x = np.load(sample_file)

    if x.ndim == 1:
        return int(x.shape[0])
    elif x.ndim == 2:
        return int(x.shape[1])
    else:
        logging.error(
            f"Expected embedding shape (D,) or (1, D), but got {x.shape}"
        )
        sys.exit(1)


def set_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
