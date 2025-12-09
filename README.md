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

## Sub-project READMEs
- `Ark+/README.md` — Ark+ evaluation instructions (uses `uv`).
- `MedImageInsights/README.md` — MedImageInsights evaluation instructions (uses `uv`).

Proceed to the sub-project README for exact commands and experiment notes.
```
