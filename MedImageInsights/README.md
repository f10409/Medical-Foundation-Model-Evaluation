# MedImageInsights

This directory contains evaluation scripts and assets for the MedImageInsights model (one of the six evaluated models).

## Installation (uv-only)
This project requires Python == 3.8.19 as specified in the `uv.lock` file.

Fetch the MedImageInsights code from the remote repo:

```bash
git clone https://huggingface.co/lion-ai/MedImageInsights
cp -rf ./MedImageInsights/2024.09.27/ ./MedImageInsights/MedImageInsight/ ./MedImageInsights/medimageinsightmodel.py ./
```

From the `FM_evaluation_public/MedImageInsights` directory, sync the environment with `uv`:

```bash
uv sync
source .venv/bin/activate
```

## Execution

Run the scripts in numeric order (prefixes like `1_0`, `1_1` indicate sequence). Example:

```bash
python 1_0_ptx_classification_experiments.py
```

Only run `.sh` wrappers when they are provided to automate intended multi-step runs.