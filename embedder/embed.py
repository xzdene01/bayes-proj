"""
Precompute ECAPA-TDNN embeddings for a subset of VoxCeleb1.

Directory layout:
    ./models/               # model cache will be stored here
    ./DATASET/              # source audio: SPK_ID/.../*.[wav, flac]
    ./embeddings/DATASET    # output embeddings mirror the source tree

!!! Always call embedder from the root of this project, so the paths are correct. !!!
The embedder part should work independently from the rest
    , it just serves to precompute the embeddings.
"""

import sys
import torch
import librosa
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Iterable

# Backend configuration, must implement download_model, load_model, encode_once
import speech_brain as backend

# Model configuration
MODELS_DIR = Path("./models")
# MODEL_ID = "spkrec-ecapa-voxceleb"
# MODEL_ID = "spkrec-resnet-voxceleb"
MODEL_ID = "spkrec-xvect-voxceleb"  # use the weakest model
LOCAL_MODEL_DIR = MODELS_DIR / MODEL_ID

# Dataset configuration

# Do NOT use VoxCeleb1 dataset with SpeechBrain models
#   , they were trained on VoxCeleb{1,2}
# DATASET = "voxceleb1_subset"
# DATASET = "VCTK"
# DATASET = "LibriSpeech"
DATASET = "CN-Celeb1"  # should be the hardes dataset here
SRC_ROOT = Path(f"./data/{DATASET}")
DST_ROOT = Path(f"./embeddings/{DATASET}_{MODEL_ID}")
TARGET_SR = 16000
AUDIO_EXTS = {".wav", ".flac"}

# Other configuration
DEVICE = "cpu"


def iter_audio_files(root: Path) -> Iterable[Path]:
    """
    Find any and all files in a given root directory that have the required
    extension, i.e., .wav or .flac.
    """
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def load_audio_mono_16k(path: Path, target_sr: int = TARGET_SR) \
        -> torch.Tensor:
    """
    Load, preprocess (resample to 16kHz + mono audio), and convert to torch
    tensor a given audio file.
    """
    y, sr = librosa.load(
        str(path),
        sr=target_sr,
        mono=True,
        dtype="float32"
    )
    return torch.from_numpy(y).unsqueeze(0)  # (1, T)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model '{MODEL_ID}' to {LOCAL_MODEL_DIR} ...")
    backend.download_model(model_id=MODEL_ID, local_dir=LOCAL_MODEL_DIR)

    print("Loading model from local path ...")
    classifier = backend.load_model(local_dir=LOCAL_MODEL_DIR, device=DEVICE)

    if not SRC_ROOT.exists():
        print(f"ERROR: Source folder not found: {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)

    files = list(iter_audio_files(SRC_ROOT))
    if not files:
        print(f"WARNING: No audio files found under {SRC_ROOT}")

    print(f"Found {len(files)} audio files. Computing embeddings ...")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for src_path in tqdm(files, unit="file"):
        # Get dst path, while keeping the directory structure
        rel = src_path.relative_to(SRC_ROOT)
        dst_path = (DST_ROOT / rel).with_suffix(".npy")
        if dst_path.exists():
            continue

        # Ensure that the parent directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Load, embed, and save the embedding of the audio file
        wav = load_audio_mono_16k(src_path)
        wav = wav.to(DEVICE)

        emb = backend.encode_once(classifier=classifier, wav=wav)

        np.save(dst_path, emb)

    print("Done. Embeddings saved under:", DST_ROOT.resolve())


if __name__ == "__main__":
    main()
