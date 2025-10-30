# BCPE - Bayesian Estimation of Prediction Uncertainty in Speaker Identification

Short, minimal project for extracting, training, and evaluating speaker embeddings.

This repository provides small utilities and training/evaluation scripts for
speaker recognition experiments using pre-trained embedders and datasets such
as CN-Celeb, LibriSpeech and VCTK.

## Key files and recommended layout

- `train.py` — training loop / fine-tuning (project-specific).
- `test.py` — evaluation script / test runs.
- `embed.py` — generate embeddings for audio files using the embedder.
- `embedder/` — embedder wrappers (e.g. SpeechBrain integrations).
- `data/` — datasets (download script `data/download.sh`).
- `embeddings/` — generated embeddings organized per dataset and model.
- `models/` — model checkpoints and architecture folders.
- `general/` — utilities: dataset loader, evaluation, model helpers, and viz.
- `runs/` — training/evaluation run outputs and logs.

## Quick start

- Pre-install:
  - Install conda on your local machine.

- Install:

```bash
conda env create -f environment.yml
conda activate bayes-proj
```

- Prepare data (choose dataset in the source code):

```bash
./data/download.sh
```

- Generate embeddings for a dataset (choose model and dataset in the source code):

```bash
python embed.py
```

- Train:

```bash
python train.py --embes EMBES_ROOT_FOLDER --out RUN_ROOT_FOLDER
```

- Evaluate:

```bash
python test.py --run RUN_ROOT_FOLDER
```

## Notes

- You can always run the training and/or testing scripts with `-h` flag to get the full list of supported arguments.
- The recommened dataset is CN-Celeb1, other supported datasets are too easy for the ECAPA-TDNN embeddings model - near 100% accuracy.
- Do not use any VoxCeleb datasets (or be very careful with them), all the SpeechBrain embedders were trained on them.
