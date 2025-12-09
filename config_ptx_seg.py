#!/usr/bin/env python
# coding: utf-8

"""
Configuration file for Fine-tuning FMs for Medical Image Segmentation
"""

class config:
    """
    Configuration class for FM Segmentation
    
    This class contains all hyperparameters and settings for training
    a medical image segmentation model using foundation models.
    """
    
    # ========================
    # Hardware Configuration
    # ========================
    GPU = 1                              # GPU device number
    WANDB_API_KEY = '1fd404a4aa7942f53225bb4b74f219f926325e2a'  # Weights & Biases API key
    
    # ========================
    # Model Architecture
    # ========================
    model_name = "google/siglip2-so400m-patch16-512"  # Pre-trained model names: "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "microsoft/rad-dino", "google/medsiglip-448","google/siglip2-so400m-patch16-512"     
    freeze_encoder = True                # Whether to freeze the encoder weights
    img_size = 512 #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
    patch_size = 16 #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
    n_channel = 1152 #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
    
    
    # ========================
    # Training Hyperparameters
    # ========================
    learning_rate = 1e-4                 # Learning rate for AdamW optimizer
    encoder_learning_rate = 1e-7         # not used if freeze_encoder
    weight_decay = 1e-2                  # Weight decay for AdamW optimizer 
    batch_size = 16                      # Training batch size
    max_epochs = 500                     # Maximum training epochs
    class_weights = [1.0, 5.0]
    
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
    msk_base_path = '/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/train_msk/'  # Base path for masks
    input_path = 'input_train_ptx_seg.csv'    # Input dataset CSV path
    mask_suffix = 'msk'
    IMG_PATH_COLUMN_NAME = 'ImagePath'
    
    # ========================
    # Experiment Tracking
    # ========================
    project = 'FM_evaluation'     # WandB project name
    test_name = 'test'             # Experiment/run name
    entity = 'f10409'                    # WandB entity/username
    
    # ========================
    # Output Configuration
    # ========================
    weight_path = f'./weights/{test_name}'  # Model weights save directory
    output_folder = f'/mnt/NAS3/projects/fli40/FM_evaluation/pred_masks/{test_name}'