import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from speechbrain.pretrained import EncoderClassifier


def download_model(model_id: str, local_dir: Path):
    snapshot_download(
        repo_id=f"speechbrain/{model_id}",
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False
    )


def load_model(local_dir: Path, device: str):
    classifier = EncoderClassifier.from_hparams(
        source=str(local_dir),
        run_opts={"device": device},
        savedir=str(local_dir)
    )
    return classifier


@torch.no_grad()
def encode_once(classifier, wav: torch.Tensor):
    emb = classifier.encode_batch(wav).squeeze(0)
    emb = emb.cpu().numpy().astype("float32")
    return emb
