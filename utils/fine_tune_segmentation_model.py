#!/usr/bin/env python
# coding: utf-8

"""
Fine-tuning FMs for Medical Image Segmentation - Function Version
"""

# Core libraries
import torch
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from models.FMSegmentator import FMSegmentator
from models.FMIntegratedSegmentator import FMIntegratedSegmentator
from utils.get_seg_dataloaders import get_seg_dataloader

# # Import configuration
# from config import Config

SEED = 5656
pl.seed_everything(SEED)


def fine_tune_segmentation_model(config=None, integrated=False):
    """
    Fine-tune a foundation model for medical image segmentation
    
    Args:
        config: Configuration object. If None, uses default Config()
    """
    
    # # Initialize configuration if not provided
    # if config is None:
    #     config = Config()
    
    # Specify which GPU to use (fallback to GPU 0 if not specified)
    gpu_device = getattr(config, 'GPU', 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_device}"
    print(f"Using GPU device: {gpu_device}")

    # Weights & Biases configuration (with fallback if not specified)
    wandb_api_key = getattr(config, 'WANDB_API_KEY', '')
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key
    os.environ['WANDB_SILENT'] = 'true'         # Suppress W&B verbose output

    # Display key configuration parameters
    print("=== Segmentation Fine-tuning Configuration ===")
    print(f"Model: {config.model_name}")
    print(f"Image size: {config.img_size}")
    print(f"Patch size: {config.patch_size}")
    if integrated:
        print(f"Patch Channels: {config.n_channel_patch}")
        print(f"CLS Channels: {config.n_channel_cls}")
    else:
        print(f"Channels: {config.n_channel}")
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

    # Display dataset overview
    print("=== Dataset Overview ===")
    print(f"Total samples: {len(input_df)}")
    print()

    # # Show first few rows of the dataset
    # print(input_df.head())

    # Create data loaders for all dataset splits
    train_dataloader, valid_dataloader, test_dataloader = get_seg_dataloader(input_df, config)

    # Display data loader information
    print("=== Data Loaders Created ===")
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(valid_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    print(f"Batch size: {config.batch_size}")

    # # Optional: Print a sample batch
    # for b in train_dataloader:
    #     print("Sample batch structure:")
    #     print(b.keys())
    #     print(f"Image shape: {b['img'].shape}")
    #     print(f"Mask shape: {b['msk'].shape}")
    #     print(f"Image range: [{b['img'].min():.3f}, {b['img'].max():.3f}]")
    #     print(f"Mask range: [{b['msk'].min():.3f}, {b['msk'].max():.3f}]")
    #     
    #     # Optional: Visualize a sample
    #     # plt.figure(figsize=(10, 5))
    #     # plt.subplot(1, 2, 1)
    #     # plt.imshow(b['img'][0,0,:,:], cmap='gray')
    #     # plt.title('Image')
    #     # plt.subplot(1, 2, 2)
    #     # plt.imshow(b['img'][0,0,:,:], cmap='gray')
    #     # plt.imshow(b['msk'][0,0,:,:], alpha=0.3, cmap='red')
    #     # plt.title('Image with Mask Overlay')
    #     # plt.show()
    #     break

    # Initialize MedSigLIP segmentation model
    if integrated:
        model = FMIntegratedSegmentator(
            config=config
        )
    else:
        model = FMSegmentator(
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
    print("=== Starting Segmentation Training ===")
    
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
    
    print("=== Segmentation Training Completed ===")
    
    # Get the best checkpoint path
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint saved at: {best_checkpoint_path}")

    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    # # Return useful objects for further analysis if needed
    # return {
    #     'model': model,
    #     'trainer': trainer,
    #     'best_checkpoint_path': best_checkpoint_path,
    #     'train_dataloader': train_dataloader,
    #     'valid_dataloader': valid_dataloader,
    #     'test_dataloader': test_dataloader,
    #     'config': config,
    #     'input_df': input_df
    # }


