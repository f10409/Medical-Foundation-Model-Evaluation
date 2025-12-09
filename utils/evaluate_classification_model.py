#!/usr/bin/env python
# coding: utf-8

"""
FM Classifier Evaluation - Function Version

This module evaluates the performance of fine-tuned FM classifiers on medical image classification tasks.
"""

# Core libraries
import torch
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# Lightning framework for model loading
import lightning as pl
from lightning.pytorch.callbacks import TQDMProgressBar

# Custom modules
from models.FMClassifier import FMClassifier
from utils.get_dataloaders import get_dataloader


# Import configuration
# from config_ptx_cla import config


def evaluate_classification_model(config=None, weight_path_override=None):
    """
    Evaluate a fine-tuned MedSigLIP classifier on test data
    
    Args:
        config: Configuration object. If None, uses Config_MedSigLIP
        GPU: GPU device number to use (default: 0)
        weight_path_override: Override the weight path from config if provided
        
    Returns:
        Dictionary containing evaluation results and predictions
    """
    
    # # Initialize configuration if not provided
    # if config is None:
    #     config = config
    
    # Specify which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.GPU}"
    print(f"Using GPU device: {config.GPU}")

    print("=== MedSigLIP Classifier Evaluation ===")
    print(f"Model: {config.model_name}")
    print(f"Classes: {config.labels}")
    print(f"Input path: {config.input_path}")
    
    # Use override weight path if provided
    if weight_path_override:
        weight_path = weight_path_override
        print(f"Using override weight path: {weight_path}")
    else:
        weight_path = config.weight_path
        print(f"Using config weight path: {weight_path}")

    # Load dataset from CSV
    print("\n=== Loading Dataset ===")
    input_df = pd.read_csv(config.input_path)
    print(f"Dataset shape: {input_df.shape}")
    print("\nDataset preview:")
    print(input_df.head())

    # Create data loader for test dataset
    print("\n=== Creating Data Loaders ===")
    _, valid_dataloader, test_dataloader = get_dataloader(input_df, config)
    print(f"Validation batches: {len(valid_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")

    # Initialize MedSigLIP classifier model
    print("\n=== Initializing Model ===")

    model = FMClassifier(
        config=config
    )
    print(f"Model initialized with config")

    # Load trained weights from checkpoint
    print("\n=== Loading Model Weights ===")
    weight_list = glob(weight_path + '/*.ckpt')
    
    if len(weight_list) > 1:
        print("Available checkpoints:")
        for i, weight in enumerate(weight_list):
            print(f"  {i}: {weight}")
        raise ValueError("There are more than one checkpoints in this folder. Please specify a single checkpoint.")
    elif len(weight_list) <= 0:
        raise ValueError(f"There is no checkpoint in this folder: {weight_path}")
    else:
        print("=== Loading Encoder Model ===")
        print(f"Checkpoint path: {weight_list[0]}")

        checkpoint = torch.load(weight_list[0], weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully")

    # Progress bar for evaluation visualization
    progress_bar = TQDMProgressBar()

    # Instantiate PyTorch Lightning trainer for evaluation
    print("\n=== Setting Up Trainer ===")
    trainer = pl.Trainer(    
        callbacks=[progress_bar],                 # Progress tracking during evaluation
        accelerator=config.accelerator,           # Use GPU acceleration
        devices=1,                                # Single GPU evaluation
        logger=False,                             # Disable logging for evaluation
        enable_checkpointing=False,               # Disable checkpointing for evaluation
    )
    print("Trainer configured for evaluation")

    # Run evaluation on test set
    print("\n=== Running Evaluation on Test Set ===")
    trainer.test(model, dataloaders=test_dataloader)
    test_predictions = model.get_test_predictions()
    
    print("=== Evaluation Completed ===")
    
    # Return results
    results = {
        'model': model,
        'trainer': trainer,
        'test_predictions': test_predictions,
        'test_dataloader': test_dataloader,
        'valid_dataloader': valid_dataloader,
        'config': config,
        'input_df': input_df,
        'checkpoint_path': weight_list[0]
    }
    
    return results


