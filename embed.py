"""
Precompute ECAPA-TDNN embeddings for a VoxCeleb dataset.

Directory layout:
    ./models/               # model cache will be stored here
    ./voxceleb1_subset/     # source audio: idXXXXX/videoID/*.wav
    ./embeddings/           # output embeddings mirror the source tree
"""

import sys
import pathlib
from typing import Iterable

import torch
import librosa
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download
from speechbrain.pretrained import EncoderClassifier

# Configuration
HF_REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
MODELS_DIR = pathlib.Path("./models")
LOCAL_MODEL_DIR = MODELS_DIR / "ecapa_voxceleb"
SRC_ROOT = pathlib.Path("./voxceleb1_subset")     # input audio root
DST_ROOT = pathlib.Path("./embeddings")           # output embeddings root
DEVICE = "cpu"                                    # device to use
TARGET_SR = 16000                                 # ECAPA expects 16 kHz
AUDIO_EXTS = {".wav", ".m4a", ".mp3"}             # should be just WAV


def iter_audio_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def load_audio_mono_16k(path: pathlib.Path, target_sr: int = TARGET_SR) \
        -> torch.Tensor:
    y, sr = librosa.load(
        str(path),
        sr=target_sr,
        mono=True,
        dtype="float32"
    )
    return torch.from_numpy(y).unsqueeze(0)  # (1, T)


def ensure_parent_dir(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    # Ensure model folder exists and download model snapshot locally
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model '{HF_REPO_ID}' to {LOCAL_MODEL_DIR} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False
    )

    # Load ECAPA-TDNN from the local folder
    print("Loading ECAPA-TDNN from local path ...")
    classifier = EncoderClassifier.from_hparams(
        source=str(LOCAL_MODEL_DIR),
        run_opts={"device": DEVICE},
        savedir=str(LOCAL_MODEL_DIR)
    )

    # Walk source tree, embed, and save .npy under ./embeddings
    if not SRC_ROOT.exists():
        print(f"ERROR: Source folder not found: {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)

    files = list(iter_audio_files(SRC_ROOT))
    if not files:
        print(f"WARNING: No audio files found under {SRC_ROOT}")

    print(f"Found {len(files)} audio files. Computing embeddings ...")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for src_path in tqdm(files, unit="file"):
        rel = src_path.relative_to(SRC_ROOT)
        dst_path = (DST_ROOT / rel).with_suffix(".npy")
        if dst_path.exists():
            continue
        ensure_parent_dir(dst_path)

        wav = load_audio_mono_16k(src_path)  # (1, T)
        wav = wav.to(DEVICE)

        with torch.no_grad():
            # EncoderClassifier.encode_batch expects (batch, time); mono is fine
            # It returns shape (batch, emb_dim); take first row
            emb = classifier.encode_batch(wav).squeeze(0).cpu().numpy().astype(np.float32)

        # Save embedding
        np.save(dst_path, emb)

    print("Done. Embeddings saved under:", DST_ROOT.resolve())


if __name__ == "__main__":
    main()
