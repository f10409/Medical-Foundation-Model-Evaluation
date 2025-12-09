#!/usr/bin/env python
# coding: utf-8

"""
Configuration file for Fine-tuning FMs for Medical Image Classification
"""

class config:
    """
    Configuration class for FM Classifier

    """
    
    # ========================
    # Label Configuration  
    # ========================
    labels = ['Pneumothorax']  # List of class labels. ex: ['No Finding', 'Pneumothorax']
    
    # ========================
    # Model Architecture
    # ========================
    model_name = "google/medsiglip-448"  # Pre-trained model names: "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "microsoft/rad-dino", "google/medsiglip-448"     
    freeze_encoder = True                # Whether to freeze the encoder weights
    
    # ========================
    # Training Hyperparameters
    # ========================
    learning_rate = 1e-4                 # Learning rate for AdamW optimizer
    encoder_learning_rate = 1e-7         # not used if freeze_encoder
    weight_decay = 1e-2                  # Weight decay for AdamW optimizer 
    batch_size = 16                      # Training batch size
    max_epochs = 500                     # Maximum training epochs
    
    # ========================
    # Training Configuration
    # ========================
    precision = 'bf16-mixed'             # Mixed precision training (bf16-mixed/16-mixed/32)
    accelerator = "gpu"                  # Training accelerator (gpu/cpu/tpu)
    num_workers = 4
    
    # ========================
    # Early Stopping & Monitoring
    # ========================
    patience = 5                        # Early stopping patience (epochs)
    min_delta = 0.0001                    # Minimum change threshold for early stopping
    
    # ========================
    # Data Configuration
    # ========================
    img_base_path = '/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/train_png/'  # Base path for images
    input_path = 'input_train_ptx_cla.csv'    # Input dataset CSV path
    
    
    # ========================
    # Experiment Tracking
    # ========================
    project = 'FM_evaluation'     # WandB project name
    test_name = 'MedSigLIP'             # Experiment/run name
    entity = 'f10409'                    # WandB entity/username
    
    # ========================
    # Environment Configuration
    # ========================
    GPU = 0  # Which GPU to use
    WANDB_API_KEY = '1fd404a4aa7942f53225bb4b74f219f926325e2a'
    
    # ========================
    # Output Configuration
    # ========================
    weight_path = f'./weights/{test_name}'  # Model weights save directory
    
    # ========================
    # Derived Configuration (Auto-computed)
    # ========================
    num_classes = len(labels)            # Number of output classes (auto-computed)
    idx_to_class = {i: label for i, label in enumerate(labels)}  # Index to class mapping (auto-computed)