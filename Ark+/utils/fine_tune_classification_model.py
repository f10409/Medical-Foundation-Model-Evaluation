#!/usr/bin/env python
# coding: utf-8

"""
Fine-tuning FMs for Medical Image Classification - Function Version
"""

# Core libraries
import torch
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import gc

# Lightning framework
import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    EarlyStopping, 
    TQDMProgressBar
)
from lightning.pytorch.loggers import WandbLogger

# Custom modules
from models.FMClassifier import FMClassifier
from utils.get_dataloaders import get_dataloader

# # Import configuration
# from config import Config
SEED = 5656
pl.seed_everything(SEED)


def fine_tune_classification_model(config=None):
    """
    Fine-tune a foundation model for medical image classification
    
    Args:
        config: Configuration object. If None, uses default Config()
    """
    
    # # Initialize configuration if not provided
    # if config is None:
    #     config = Config()
    
    # Specify which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.GPU}"
    print(f"Using GPU device: {config.GPU}")

    # Weights & Biases configuration
    os.environ['WANDB_API_KEY'] = config.WANDB_API_KEY
    os.environ['WANDB_SILENT'] = 'true'         # Suppress W&B verbose output

    # Display key configuration parameters
    print("=== Fine-tuning Configuration ===")
    print(f"Model: {config.model_name}")
    print(f"Classes: {config.labels}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Freeze encoder: {config.freeze_encoder}")
    print(f"Experiment: {config.project}/{config.test_name}")

    # Load dataset from CSV
    input_df = pd.read_csv(config.input_path)

    # Display dataset split distribution
    print("=== Dataset Split Distribution ===")
    print(input_df.Split.value_counts())
    print()

    # Display class label distribution
    print("=== Class Label Distribution ===")
    print(input_df[config.labels[0]].value_counts())
    print()

    # Display dataset overview
    print("=== Dataset Overview ===")
    print(f"Total samples: {len(input_df)}")
    print()

    # # Show first few rows of the dataset
    # print(input_df.head())

    # Create data loaders for all dataset splits
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(input_df, config)

    # Display data loader information
    print("=== Data Loaders Created ===")
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(valid_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    print(f"Batch size: {config.batch_size}")

    # # Optional: Print a sample batch
    # for b in train_dataloader:
    #     print("Sample batch structure:")
    #     print(b)
    #     print(f"Image shape: {b['img'].shape}")
    #     break

    # Initialize MedSigLIP classifier model
    model = FMClassifier(
        config=config
    )

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Model checkpointing - save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.weight_path}",
        filename=f'{config.test_name}_{{epoch}}_{{val_loss:0.4F}}',
        monitor="val_loss",
        mode="min",
        save_last=False,
        save_top_k=1
    )

    # Early stopping to prevent overfitting
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=config.min_delta,
        patience=config.patience,
        verbose=False,
        mode='min'
    )

    # Weights & Biases logger for experiment tracking
    wandb_logger = WandbLogger(
        save_dir=f"{config.weight_path}",
        name=f'{config.test_name}',
        project=config.project,
        entity=config.entity,
        offline=False,
        log_model=False,
        config={"Creator": "HITI"}
    )

    # Progress bar for training visualization
    progress_bar = TQDMProgressBar()

    # Display callback configuration
    print("=== Training Configuration ===")
    print(f"Checkpoint directory: {config.weight_path}")
    print(f"Early stopping patience: {config.patience} epochs")
    print(f"Min delta for early stopping: {config.min_delta}")
    print(f"W&B project: {config.project}")
    print(f"Experiment name: {config.test_name}")

    # Instantiate PyTorch Lightning trainer
    trainer = pl.Trainer(
        gradient_clip_val=1.0,                    # Clip gradients to prevent exploding gradients
        callbacks=[progress_bar, lr_monitor, checkpoint_callback, early_stop_callback],
        logger=wandb_logger,                      # W&B logger for experiment tracking
        precision=config.precision,               # Mixed precision training (bf16-mixed)
        accelerator=config.accelerator,           # Use GPU acceleration
        devices=1,                                # Single GPU training
        log_every_n_steps=1,                      # Log metrics every step
        default_root_dir=config.weight_path,      # Directory for trainer outputs
        max_epochs=config.max_epochs              # Maximum training epochs
    )

    # Train the model
    print("=== Starting Training ===")
    
    # Uncomment if needed for MONAI compatibility
    # from monai.data.meta_tensor import MetaTensor
    # torch.serialization.add_safe_globals([
    #     np.core.multiarray._reconstruct,
    #     np.ndarray,
    #     np.dtype,
    #     np.core.multiarray.scalar,
    #     MetaTensor,
    # ])

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("=== Training Completed ===")
    
    # Return useful objects for further analysis if needed
    # return {
    #     'model': model,
    #     'trainer': trainer,
    #     'train_dataloader': train_dataloader,
    #     'valid_dataloader': valid_dataloader,
    #     'test_dataloader': test_dataloader,
    #     'config': config,
    #     'input_df': input_df
    # }


