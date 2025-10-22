import torch
import numpy as np

from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, files: List[Path],
                 embeddings_root: Path,
                 label_map: Dict[str, int]):
        self.files = files
        self.embeddings_root = embeddings_root
        self.label_map = label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        # Load embedding and reshape (1, D) -> (D,)
        emb = np.load(p).astype(np.float32).squeeze(0)
        x = torch.from_numpy(emb)

        # Extract speaker folder name and get SpkID from label map
        spk = p.relative_to(self.embeddings_root).parts[0]
        y = self.label_map[spk]
        return x, y
