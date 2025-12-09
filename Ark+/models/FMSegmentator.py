import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
# from transformers import SiglipVisionModel, AutoModel 
# from open_clip import create_model_from_pretrained
from timm.models.swin_transformer import SwinTransformer

import numpy as np
import os
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score

class LinearProbe(nn.Module):
    def __init__(self, num_classes=1, size=518, n_channel=768):
        super().__init__()
        self.conv = nn.Conv2d(n_channel, num_classes, kernel_size=1, stride=1)
        self.size = size
        
    def forward(self, x):
        x = self.conv(x)  # [B, num_classes, 37, 37]
        
        x = nn.functional.interpolate(
            x, 
            size=(self.size, self.size), 
            mode='bilinear',
            align_corners=True
        )  # [B, num_classes, 518, 518]
        return x


class FMSegmentator(pl.LightningModule):
    """Lightning Module for Foundation Model segmentation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freeze_encoder = config.freeze_encoder
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights))
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create output directories if they don't exist
        self._create_output_folderectories()
        
        # Load pre-trained vision encoder based on model name
        # if config.model_name == "google/medsiglip-448":
        #     self.vision_encoder = SiglipVisionModel.from_pretrained(config.model_name)
            
        # if config.model_name == "microsoft/rad-dino":
        #     self.vision_encoder = AutoModel.from_pretrained(config.model_name)  
            
        # if config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli":    
        #     self.vision_encoder = AutoModel.from_pretrained(config.model_name).vision_model 
            
        # if config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 
        #     self.vision_encoder, _ = create_model_from_pretrained(config.model_name)
            
        # if config.model_name == "google/siglip2-so400m-patch16-512":
        #     self.vision_encoder = SiglipVisionModel.from_pretrained(config.model_name)

        # if config.model_name == "facebook/dinov2-base":
        #     self.vision_encoder = AutoModel.from_pretrained(config.model_name) 
        
        # if config.model_name == "MedImageInsights":
        #     # Initialize classifier
        #     classifier = MedImageInsight(model_dir="2024.09.27", vision_model_name="medimageinsigt-v1.0.0.pt", language_model_name="language_model.pth")
        #     # Load model
        #     classifier.load_model()
        #     self.vision_encoder = classifier.model.image_encoder

        if config.model_name == "Ark+":
            # Initialize the model
            model = SwinTransformer(
                num_classes=0,
                img_size=768,
                patch_size=4,
                window_size=12,
                embed_dim=192,
                depths=(2, 2, 18, 2),
                num_heads=(6, 12, 24, 48), use_checkpoint=True
            )
            
            # Load the checkpoint
            checkpoint = torch.load('./Ark6_swinLarge768_ep50.pth.tar', map_location="cpu", weights_only=False)
            state_dict = checkpoint['teacher']
            
            # Remove "module." prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # Identify and delete unnecessary keys
            k_del = [k for k in state_dict.keys() if "attn_mask" in k] + ['head.weight', 'head.bias']
            print(f"Removing key(s) {k_del} from pretrained checkpoint for scaled input size")
            
            # Delete identified keys
            for k in k_del:
                if k in state_dict:  # Ensure the key exists
                    del state_dict[k]
            
            # Load the model weights
            msg = model.load_state_dict(state_dict, strict=False)
            print('Loaded with msg:', msg)

            self.vision_encoder = model
        
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()  # Set to eval mode when frozen
        else:
            # Ensure encoder is in training mode when not frozen
            self.vision_encoder.train()
        
        self.segmentator = LinearProbe(size=config.img_size, num_classes=2, n_channel=config.n_channel)
        
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
            print(self.config.test_name)
            print(f"  Predictions: {pred_dir}")
            print(f"  Probabilities: {prob_dir}")
            print(f"  Ground Truth: {gt_dir}")

    
    def reshape_patch_embeddings(self, flat_tokens: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
        """Reshape flat list of patch tokens into a spatial grid."""
        h = w = image_size // patch_size
        b, _, c = flat_tokens.shape
        return flat_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def extract_with_hook(self, model, x):
        patch_embeddings = []
        
        def hook_fn(module, input, output):
            patch_embeddings.append(output.clone())
        
        # Register hook on patch_embed layer
        if self.config.patch_size == 32:
            handle = model.norm.register_forward_hook(hook_fn)
        if self.config.patch_size == 16:
            handle = model.layers[2].blocks[17].register_forward_hook(hook_fn)
        if self.config.patch_size == 8:
            handle = model.layers[1].blocks[1].register_forward_hook(hook_fn)
        if self.config.patch_size == 4:
            handle = model.layers[0].blocks[1].register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Remove hook
        handle.remove()
        
        return patch_embeddings[0]
    
    def forward(self, pixel_values):
        # Extract patch embeddings from vision encoder
        # if self.config.model_name == "microsoft/rad-dino":
        #     vision_outputs = self.vision_encoder(pixel_values)
        #     flat_patch_embeddings = vision_outputs.last_hidden_state[:, 1:]  # Remove CLS token
        # if self.config.model_name == "google/medsiglip-448":
        #     vision_outputs = self.vision_encoder(pixel_values)
        #     flat_patch_embeddings = vision_outputs.last_hidden_state
        # if self.config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli": 
        #     vision_outputs = self.vision_encoder(pixel_values)
        #     flat_patch_embeddings = vision_outputs.last_hidden_state
        # if self.config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        #     vision_outputs = self.vision_encoder.visual.trunk.forward_features(pixel_values)
        #     flat_patch_embeddings = vision_outputs[:, 1:, :]            
        # if self.config.model_name == "google/siglip2-so400m-patch16-512":
        #     vision_outputs = self.vision_encoder(pixel_values)
        #     flat_patch_embeddings = vision_outputs.last_hidden_state
        # if self.config.model_name == "facebook/dinov2-base":
        #     vision_outputs = self.vision_encoder(pixel_values)
        #     flat_patch_embeddings = vision_outputs.last_hidden_state[:, 1:]  # Remove CLS token
        
        # if self.config.model_name == "MedImageInsights":
        #     vision_outputs = self.vision_encoder.forward_patch_features(pixel_values)  
        #     flat_patch_embeddings = vision_outputs[2]

        if self.config.model_name == "Ark+":
            flat_patch_embeddings = self.extract_with_hook(self.vision_encoder, pixel_values)

        
        # Reshape patch embeddings to spatial format
        logits = self.reshape_patch_embeddings(flat_patch_embeddings, 
                                               image_size=self.config.img_size, 
                                               patch_size=self.config.patch_size)
        
        # Apply segmentation head
        logits = self.segmentator(logits)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        # Extract data from batch
        pixel_values = batch['img']
        masks = batch['msk']
        masks = masks[:,0:1,:,:]
        masks = torch.cat((1 - masks, masks), dim=1)
        #print(masks.shape)
        
        # Forward pass to get predicted masks
        pred_masks = self(pixel_values)
        #print(pred_masks.shape)
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
        #print(masks)
        masks = masks[:,0:1,:,:]
        masks = torch.cat((1 - masks, masks), dim=1)
        
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
        masks = torch.cat((1 - masks, masks), dim=1)
        
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
            mask = (pred_masks[i].argmax(dim=0).float().cpu().numpy())
            #print('pred', mask.shape, mask.mean())
            mask_prob = pred_masks[i,1,:,:].sigmoid().float().cpu().numpy()
            #print('prob', mask_prob.shape, mask_prob.mean())
            g = (masks[i,1,:,:].float().cpu().numpy()).astype(int)
            #print('gt', g.shape, g.mean())

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