import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
from transformers import SiglipVisionModel, AutoModel 
from open_clip import create_model_from_pretrained

import numpy as np
import os
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score


class IntegratedSeg(nn.Module):
    def __init__(self, num_classes, output_shape, input_grid_size=37,
                 feature_dim_patch=768, feature_dim_cls=768, refined_dim=256, num_heads=8):
        """
        Args:
            num_classes (int): Number of segmentation classes.
            output_shape (tuple): Desired output shape of segmentation map (H, W).
            input_grid_size (int): Size of the patch grid (e.g., 37 for 37x37).
            feature_dim (int): Input feature dimension (e.g., 768 for DINOv2 ViT-S).
            refined_dim (int): Output dimension of CNN refiner.
            num_heads (int): Number of attention heads in cross-attention.
        """
        super().__init__()
        self.feature_dim_patch = feature_dim_patch
        self.feature_dim_cls = feature_dim_cls
        self.refined_dim = refined_dim
        self.input_grid_size = input_grid_size  # e.g., 37 for 37x37 grid
        self.output_shape = output_shape
        
        # Added Layer Normalization before CNN refiner
        self.ln_before_cnn = nn.LayerNorm([feature_dim_patch, input_grid_size, input_grid_size])
        
        # CNN Refiner
        self.cnn_refiner = nn.Sequential(
            nn.Conv2d(feature_dim_patch, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, refined_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(refined_dim),
            nn.ReLU(inplace=True),
        )

        self.cnn_kv_proj = nn.Conv2d(refined_dim, refined_dim*2, kernel_size=1)
        
        # CLS token LayerNorm
        self.cls_ln = nn.LayerNorm(feature_dim_cls)

        # Cross-Attention with CLS token
        self.cls_proj = nn.Conv2d(feature_dim_cls, refined_dim, kernel_size=1)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=refined_dim, num_heads=num_heads, batch_first=True)
                
        # Segmentation Head
        self.seg_head = nn.Conv2d(refined_dim, num_classes, kernel_size=1)
        
    def forward(self, cls_token, patch_tokens):
        """
        Args:
            cls_token (torch.Tensor): CLS token [B, feature_dim], e.g., [B, 768]
            patch_tokens (torch.Tensor): Reshaped patch tokens [B, feature_dim, H_grid, W_grid],
                                        e.g., [B, 768, 37, 37]
        Returns:
            torch.Tensor: Segmentation logits [B, num_classes, H, W]
        """
        B = patch_tokens.shape[0]
                
        # Apply Layer Norm after CNN
        patch_tokens = self.ln_before_cnn(patch_tokens)

        # Upsample patch tokens to intermediate size for CNN processing
        patch_tokens = F.interpolate(patch_tokens, size=(64, 64), mode='bilinear',
                                       align_corners=False)  # [B, 768, 64, 64]
        
        # Refine with CNN
        patch_tokens = self.cnn_refiner(patch_tokens)  # [B, 256, 64, 64]
                
        # Create K and V from CNN refined patch tokens using projection layer
        kv = self.cnn_kv_proj(patch_tokens)  # [B, refined_dim*2, 64, 64]
        K, V = torch.chunk(kv, 2, dim=1)      # Each: [B, refined_dim, 64, 64]
        H, W = K.shape[2:]
        K = K.view(B, self.refined_dim, -1).permute(0, 2, 1)  # [B, H*W, refined_dim]
        V = V.view(B, self.refined_dim, -1).permute(0, 2, 1)  # [B, H*W, refined_dim]

        # Process CLS token to create Q vector
        Q = self.cls_ln(cls_token)
        Q = Q.unsqueeze(-1).unsqueeze(-1)                # [B, feature_dim, 1, 1]
        Q = self.cls_proj(Q)                             # [B, refined_dim, 1, 1]
        Q = Q.view(B, self.refined_dim, 1).permute(0, 2, 1)  # [B, 1, refined_dim]

        # Apply cross-attention: Q attends to K and V
        attn_output, _ = self.cross_attn(query=Q, key=K, value=V)  # [B, 1, refined_dim]

        # Expand the global context across spatial dimensions and fuse with patch tokens
        global_context = attn_output.expand(-1, H*W, -1)          # [B, H*W, refined_dim]
        global_context = global_context.permute(0, 2, 1).view(B, self.refined_dim, H, W)  # [B, refined_dim, H, W]
        attn_features = patch_tokens + global_context   # Residual connection

        attn_features = F.interpolate(attn_features, size=self.output_shape, mode='bilinear', align_corners=False)
        
        # Segmentation head
        logits = self.seg_head(attn_features)  # [B, num_classes, *output_shape]
        
        return logits


class FMIntegratedSegmentator(pl.LightningModule):
    """Lightning Module for Foundation Model segmentation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freeze_encoder = config.freeze_encoder
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create output directories if they don't exist
        self._create_output_folderectories()
        
        # Load pre-trained vision encoder based on model name
        if config.model_name == "google/medsiglip-448":
            self.vision_encoder = SiglipVisionModel.from_pretrained(config.model_name)
            
        if config.model_name == "microsoft/rad-dino":
            self.vision_encoder = AutoModel.from_pretrained(config.model_name)  
            
        if config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli":    
            self.vision_encoder = AutoModel.from_pretrained(config.model_name).vision_model 
            
        if config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 
            self.vision_encoder, _ = create_model_from_pretrained(config.model_name)
            
        if config.model_name == "google/siglip2-so400m-patch16-512":
            self.vision_encoder = SiglipVisionModel.from_pretrained(config.model_name)
             
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()  # Set to eval mode when frozen
        else:
            # Ensure encoder is in training mode when not frozen
            self.vision_encoder.train()
        
        self.segmentator = IntegratedSeg(1, (config.img_size, config.img_size), 
                                         input_grid_size=config.img_size//config.patch_size, 
                                         feature_dim_patch=config.n_channel_patch, feature_dim_cls=config.n_channel_cls)
        
        # Metrics storage for validation and test
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _create_output_folderectories(self):
        """Create output directories for saving predictions, probabilities, and ground truth."""
        if hasattr(self.config, 'output_folder'):
            output_folder = self.config.output_folder
            
            # Create main output directory
            os.makedirs(output_folder, exist_ok=True)
            
            # Create subdirectories for different outputs
            pred_dir = os.path.join(output_folder, 'Pred')
            prob_dir = os.path.join(output_folder, 'Prob')
            gt_dir = os.path.join(output_folder, 'GT')
            
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(prob_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            
            print(f"Created output directories:")
            print(f"  Predictions: {pred_dir}")
            print(f"  Probabilities: {prob_dir}")
            print(f"  Ground Truth: {gt_dir}")

    
    def reshape_patch_embeddings(self, flat_tokens: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
        """Reshape flat list of patch tokens into a spatial grid."""
        h = w = image_size // patch_size
        b, _, c = flat_tokens.shape
        return flat_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
    
    def forward(self, pixel_values):
        # Extract patch embeddings from vision encoder
        if self.config.model_name == "microsoft/rad-dino":
            vision_outputs = self.vision_encoder(pixel_values)
            flat_patch_embeddings = vision_outputs.last_hidden_state[:, 1:]  # Remove CLS token
            pooled_output = vision_outputs.pooler_output
        if self.config.model_name == "google/medsiglip-448":
            vision_outputs = self.vision_encoder(pixel_values)
            flat_patch_embeddings = vision_outputs.last_hidden_state
            pooled_output = vision_outputs.pooler_output
        if self.config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli": 
            vision_outputs = self.vision_encoder(pixel_values)
            flat_patch_embeddings = vision_outputs.last_hidden_state
            pooled_output = vision_outputs.pooler_output
        if self.config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            vision_outputs = self.vision_encoder.visual.trunk.forward_features(pixel_values)
            flat_patch_embeddings = vision_outputs[:, 1:, :]    
            pooled_output = self.vision_encoder.visual(pixel_values)
        if self.config.model_name == "google/siglip2-so400m-patch16-512":
            vision_outputs = self.vision_encoder(pixel_values)
            flat_patch_embeddings = vision_outputs.last_hidden_state
            pooled_output = vision_outputs.pooler_output

        
        # Reshape patch embeddings to spatial format
        logits = self.reshape_patch_embeddings(flat_patch_embeddings, 
                                               image_size=self.config.img_size, 
                                               patch_size=self.config.patch_size)
        
        # Apply segmentation head
        logits = self.segmentator(pooled_output, logits)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        # Extract data from batch
        pixel_values = batch['img']
        masks = batch['msk']
        masks = masks[:,0:1,:,:]
        
        # Forward pass to get predicted masks
        pred_masks = self(pixel_values)
        loss = self.criterion(pred_masks, masks)

        # Log training loss
        self.log('train_loss', loss, prog_bar=True, 
               batch_size=self.config.batch_size, 
               on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract data from batch
        pixel_values = batch['img']
        masks = batch['msk']
        masks = masks[:,0:1,:,:]
        
        # Forward pass to get predicted masks
        pred_masks = self(pixel_values)
        loss = self.criterion(pred_masks, masks)

        # Log validation loss
        self.log('val_loss', loss, prog_bar=True, 
               batch_size=self.config.batch_size, 
               on_step=False, on_epoch=True)
          
        return loss
    
    def test_step(self, batch, batch_idx):
        # Extract data from batch
        pixel_values = batch['img']
        masks = batch['msk']
        masks = masks[:,0:1,:,:]
        paths = batch['paths']
        
        # Forward pass to get predicted masks
        pred_masks = self(pixel_values)
        loss = self.criterion(pred_masks, masks)

        # Log test loss
        self.log('test_loss', loss, prog_bar=True, 
               batch_size=self.config.batch_size, 
               on_step=False, on_epoch=True)

        # Calculate IoU and Dice scores for each sample in batch
        ious = []
        dices = []
        for i, p_ in enumerate(paths):
            fn = p_.split('/')[-1][:-4]  # Extract filename without extension
            
            # Convert predictions to binary masks and probabilities
            mask = (pred_masks[i].sigmoid().float().cpu().numpy() > 0.5).astype(int)
            mask_prob = pred_masks[i].sigmoid().float().cpu().numpy()
            g = (masks[i].float().cpu().numpy()).astype(int)

            # Flatten for metric calculation
            g_flat = g.flatten()
            mask_flat = mask.flatten()
            
            # Calculate IoU (Jaccard) and Dice (F1) scores
            iou = jaccard_score(g_flat, mask_flat)
            dice = f1_score(g_flat, mask_flat)

            ious.append(iou)
            dices.append(dice)

            # Save predicted masks, probabilities, and ground truth
            mask = (mask * 255).squeeze().astype(np.uint8) 
            mask_img = Image.fromarray(mask)             
            mask_img.save(f'{self.config.output_folder}/Pred/{fn}.png')  # Save as grayscale

            mask_prob = (mask_prob * 255).squeeze().astype(np.uint8) 
            mask_prob_img = Image.fromarray(mask_prob)             
            mask_prob_img.save(f'{self.config.output_folder}/Prob/{fn}.png')  # Save as grayscale

            g = (g * 255).squeeze().astype(np.uint8)  
            g_img = Image.fromarray(g) 
            g_img.save(f'{self.config.output_folder}/GT/{fn}.png')  # Save as grayscale

        # Store outputs for final evaluation
        self.test_step_outputs.append({
            'iou': ious,
            'dice': dices,
            'paths': paths
        })
        
        # Return metrics for this batch
        return {'iou': ious,
            'dice': dices,
            'paths': paths}
        

    def get_test_predictions(self):
        """
        Get all test predictions, IoU and Dice scores, and paths.
        Should be called after test_step but before on_test_epoch_end.
        """
        if not self.test_step_outputs:
            return None
            
        # Concatenate all IoU and Dice scores from test steps
        all_ious = np.concatenate([x['iou'] for x in self.test_step_outputs])
        all_dices = np.concatenate([x['dice'] for x in self.test_step_outputs])
        # Flatten list of paths
        all_paths = [x['paths'] for x in self.test_step_outputs]
        all_paths = [item for sublist in all_paths for item in sublist]
        
        return {
            'iou': all_ious,
            'dice': all_dices, 
            'paths': all_paths  # Fixed typo: was 'all_pathsx'
        }
        
    
    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        if not self.freeze_encoder:
            # Differentiated learning rates: lower for encoder, higher for segmentator
            encoder_lr = getattr(self.config, 'encoder_learning_rate', self.config.learning_rate * 0.1)
            segmentator_lr = self.config.learning_rate  # Fixed typo: was 'csegmentator_lr'
            
            param_groups = [
                {
                    'params': self.vision_encoder.parameters(),
                    'lr': encoder_lr,
                    'name': 'encoder'
                },
                {
                    'params': self.segmentator.parameters(),  # Fixed: was 'classifier'
                    'lr': segmentator_lr,
                    'name': 'segmentator'  # Fixed: was 'classifier'
                }
            ]
            
            print(f"Using differentiated learning rates:")
            print(f"  Encoder LR: {encoder_lr}")
            print(f"  Segmentator LR: {segmentator_lr}")  # Fixed: was 'classifier_lr'
            
        else:
            # If encoder is frozen, only optimize segmentator parameters
            param_groups = [
                {
                    'params': self.segmentator.parameters(),  # Fixed: was 'classifier'
                    'lr': self.config.learning_rate,
                    'name': 'segmentator'  # Fixed: was 'classifier'
                }
            ]
            print(f"Encoder frozen. Using single LR: {self.config.learning_rate}")
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
        return {
            'optimizer': optimizer
        }