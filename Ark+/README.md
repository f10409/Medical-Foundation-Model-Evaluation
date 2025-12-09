# Ark+

This directory contains the evaluation scripts and assets for the Ark+ model.

## Important pre-step (repository root)
- Before running per-model evaluations, run `uv sync` from the repository root to ensure packages are available:

```bash
uv sync
```

## Installation (uv-only)
1. From the repository root, change into the Ark+ directory:

```bash
cd Ark+
```

2. Sync the Ark+ environment (creates/updates `.venv` according to `uv.lock`):

```bash
uv sync
```

3. Activate the environment created by `uv`:

```bash
source .venv/bin/activate
```
4. Follow the Instructions to download the weight to this folder: https://github.com/jlianglab/Ark

## Execution

Run the scripts in numeric order (prefixes like `1_0`, `1_1` indicate sequence). Example:

```bash
python 1_0_ptx_classification_experiments.py
```

Only run `.sh` wrappers when they are present.
```