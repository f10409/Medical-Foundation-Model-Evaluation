## Overview: running the model evaluations
Repository for evaluating medical foundation models. This repository contains the code and scripts to evaluate eight different foundation models. Two subfolders contain model-specific evaluation pipelines:

- `Ark+/` — evaluation scripts and assets for the Ark+ model evaluations.
- `MedImageInsights/` — evaluation scripts and assets for the MedImageInsights model evaluations.

Important: this repository standardizes on `uv` for environment creation and dependency installation. All instructions below and in sub-project READMEs assume you will use `uv`.

## Quick start
1. From the repository root, sync the shared environment and tools used by the evaluations of six models (BiomedCLIP, CheXagent, MedSigLIP, RAD-DINO, DINOv2, SigLIP2):

```bash
uv sync
```

2. Activate the environment created by `uv`:

```bash
source .venv/bin/activate
```

3. Run the experiment scripts in numeric order (file prefixes like `1_0`, `1_1` denote sequence). Example:

```bash
python 1_0_ptx_classification_experiments.py
```

4. When finished, exit the environment with:

```bash
deactivate
```

Notes:
- Only run accompanying `.sh` wrappers when they are present to automate steps (they typically exist alongside `.py` files with the same numeric prefix).
- Check a sub-project's `uv.lock` for the expected Python version (for example, `MedImageInsights` may require Python 3.8).

## Data

This repository expects local copies of the datasets used for evaluation. Below are brief instructions for preparing two commonly used datasets.

- SIIM-ACR Pneumothorax
	- Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
	- Preparation steps:
		1. Download the DICOM files from the Kaggle competition.
		2. Convert DICOMs to PNGs (keep the same base filename) and save them into `{YOUR PATH}/train_png/`.		 
		3. Convert the RLE masks provided by the competition to binary PNG masks (same base filename) and save them into `{YOUR PATH}/train_msk/`.
        4. The `inputs` folder contains the files for 5-fold cross-validation. Each file (e.g., `input_train_ptx_cla_0.csv` for the first fold) includes the image paths and labels for that specific fold.
        5. The same folder contains `ptx_volume_pct.csv`, which records the pneumothorax volumes.

- EmoryCXR
	- This dataset cannot be publicly distributed from this repository. To request access, please contact Dr. Judy Gichoya: judywawira@emory.edu

## Sub-project READMEs
- `Ark+/README.md` — Ark+ evaluation instructions (uses `uv`).
- `MedImageInsights/README.md` — MedImageInsights evaluation instructions (uses `uv`).

Proceed to the sub-project README for exact commands and experiment notes.
```
