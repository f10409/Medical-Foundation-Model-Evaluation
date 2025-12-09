import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
from transformers import SiglipVisionModel, AutoModel, SegformerForImageClassification 
from open_clip import create_model_from_pretrained
from sklearn.metrics import roc_auc_score

class FMClassifier(pl.LightningModule):
    """Lightning Module for FM classification"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freeze_encoder = config.freeze_encoder
        self.idx_to_class = config.idx_to_class  # Changed parameter name
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Load pre-trained vision encoder
        if config.model_name == "google/medsiglip-448":
            self.vision_encoder = SiglipVisionModel.from_pretrained(config.model_name)
            self.embed_dim = 1152 #self.vision_encoder.config.hidden_size
        if config.model_name == "microsoft/rad-dino":
            self.vision_encoder = AutoModel.from_pretrained(config.model_name)  
            self.embed_dim = 768
        if config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli":    
            self.vision_encoder = AutoModel.from_pretrained(config.model_name).vision_model 
            self.embed_dim = 1024
        if config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 
            self.vision_encoder, _ = create_model_from_pretrained(config.model_name)
            self.embed_dim = 512
        if config.model_name == "google/siglip2-so400m-patch16-512":
            self.vision_encoder = SiglipVisionModel.from_pretrained(config.model_name)
            self.embed_dim = 1152 
        if config.model_name == "facebook/dinov2-base":
            self.vision_encoder = AutoModel.from_pretrained(config.model_name)  
            self.embed_dim = 768
        if config.model_name == "nvidia/segformer-b4-finetuned-ade-512-512": 
            self.vision_encoder = SegformerForImageClassification.from_pretrained(config.model_name, num_labels=config.num_classes, ignore_mismatched_sizes=True)
            self.config.encoder_learning_rate = 0.000001
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()  # Set to eval mode when frozen
        else:
            # Ensure encoder is in training mode when not frozen
            self.vision_encoder.train()
        
        # Get embedding dimension
        # self.embed_dim = self.vision_encoder.config.hidden_size
        
        # Classification head
        # self.classifier = nn.Sequential(
        #     nn.Dropout(config.dropout_rate),
        #     nn.Linear(self.embed_dim, config.hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(config.dropout_rate),
        #     nn.Linear(config.hidden_dim, config.num_classes)
        # )

        if config.model_name == "nvidia/segformer-b4-finetuned-ade-512-512":
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, config.num_classes)
            )
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, pixel_values):
        # Extract features from vision encoder
        if self.config.model_name == "google/medsiglip-448":
            vision_outputs = self.vision_encoder(pixel_values)
            pooled_output = vision_outputs.pooler_output
        if self.config.model_name == "microsoft/rad-dino":
            vision_outputs = self.vision_encoder(pixel_values)
            pooled_output = vision_outputs.pooler_output
        if self.config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli": 
            vision_outputs = self.vision_encoder(pixel_values)
            pooled_output = vision_outputs.pooler_output
        if self.config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
            pooled_output = self.vision_encoder.visual(pixel_values)
        if self.config.model_name == "google/siglip2-so400m-patch16-512":
            vision_outputs = self.vision_encoder(pixel_values)
            pooled_output = vision_outputs.pooler_output 
        if self.config.model_name == "facebook/dinov2-base":
            vision_outputs = self.vision_encoder(pixel_values)
            pooled_output = vision_outputs.pooler_output 
        if self.config.model_name == "nvidia/segformer-b4-finetuned-ade-512-512":
            pooled_output = self.vision_encoder(pixel_values)
            pooled_output = pooled_output.logits
        # Apply classification head
        logits = self.classifier(pooled_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch['img'], batch['cls']
        logits = self(pixel_values)

        # if len(label.shape)==1:
        #     label = torch.unsqueeze(label, dim=-1)
            
        # Use BCEWithLogitsLoss for multi-label classification
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.config.batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch['img'], batch['cls']
        logits = self(pixel_values)
        
        # if len(label.shape)==1:
        #     label = torch.unsqueeze(label, dim=-1)
            
        # Use BCEWithLogitsLoss for consistency
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Store for epoch-end calculations
        self.validation_step_outputs.append({
            'val_loss': loss,
            'logits': logits,
            'labels': labels
        })
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.config.batch_size)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate epoch-level metrics
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Calculate predictions
        all_probs = torch.sigmoid(all_logits)
        num_classes = all_probs.shape[1]

        # # Debug prints
        # print(f"Labels shape: {all_labels.shape}")
        # print(f"Probs shape: {all_probs.shape}")
        # print(f"Number of classes: {num_classes}")
        # print(f"Unique labels: {torch.unique(all_labels)}")
        
        # Calculate per-class AUC metrics
        for c in range(num_classes):
            # Per-class AUC score
            try:
                auc = roc_auc_score(
                    all_labels[:, c].detach().cpu().float().numpy(),
                    all_probs[:, c].detach().cpu().float().numpy()
                )
                self.log(f'val_auc_{self.idx_to_class[c]}', auc, on_epoch=True, prog_bar=True, batch_size=self.config.batch_size)
            except ValueError:
                # Skip logging AUC if only one class is present
                pass
        
        # Only calculate macro AUC if there are multiple classes
        if num_classes > 1:
            # Calculate macro AUC
            try:
                macro_auc = roc_auc_score(
                    all_labels.detach().cpu().float().numpy(),
                    all_probs.detach().cpu().float().numpy(),
                    average='macro'
                )
                self.log('val_macro_auc', macro_auc, on_epoch=True, prog_bar=True, batch_size=self.config.batch_size)
            except ValueError:
                # Skip logging macro AUC if calculation fails
                pass
        
        # Clear stored outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        pixel_values, labels, paths = batch['img'], batch['cls'], batch['paths']
        logits = self(pixel_values)
    
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        
        # Store for final evaluation
        self.test_step_outputs.append({
            'test_loss': loss,
            'logits': logits,
            'labels': labels,
            'probs': probs,
            'paths': paths
        })
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=self.config.batch_size)
        
        # Return dictionary with loss, probabilities, and labels
        return {
            'loss': loss,
            'probs': probs,
            'labels': labels,
            'logits': logits,
            'paths': paths
        }

    
    def get_test_predictions(self):
        """
        Get all test predictions, probabilities, and labels.
        Should be called after test_step but before on_test_epoch_end.
        """
        if not self.test_step_outputs:
            return None
            
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        all_paths = [x['paths'] for x in self.test_step_outputs]
        all_paths = [item for sublist in all_paths for item in sublist]
        #all_preds = (all_probs >= 0.5).float()
        
        return {
            'logits': all_logits.cpu().detach().numpy(),
            'labels': all_labels.cpu().detach().numpy(), 
            'probs': all_probs.cpu().detach().numpy(),
            'paths': all_paths,
            #'preds': all_preds
        }
    
    def on_test_epoch_end(self):
        
        # Calculate final test metrics using stored probabilities
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])  # Use stored probs
        num_classes = all_probs.shape[1]
        
        # Calculate per-class AUC metrics
        for c in range(num_classes):
            # Per-class AUC score
            try:
                auc = roc_auc_score(
                    all_labels[:, c].detach().cpu().float().numpy(),
                    all_probs[:, c].detach().cpu().float().numpy()
                )
                self.log(f'test_auc_{self.idx_to_class[c]}', auc, on_epoch=True, batch_size=self.config.batch_size)
            except ValueError:
                # Skip logging AUC if only one class is present
                pass
        
        # Only calculate macro AUC if there are multiple classes
        if num_classes > 1:
            # Calculate macro AUC
            try:
                macro_auc = roc_auc_score(
                    all_labels.detach().cpu().float().numpy(),
                    all_probs.detach().cpu().float().numpy(),
                    average='macro'
                )
                self.log('test_macro_auc', macro_auc, on_epoch=True, batch_size=self.config.batch_size)
            except ValueError:
                # Skip logging macro AUC if calculation fails
                pass
        
    
    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        if not self.freeze_encoder:
            # Differentiated learning rates: lower for encoder, higher for classifier
            encoder_lr = getattr(self.config, 'encoder_learning_rate', self.config.learning_rate * 0.1)
            classifier_lr = self.config.learning_rate
            
            param_groups = [
                {
                    'params': self.vision_encoder.parameters(),
                    'lr': encoder_lr,
                    'name': 'encoder'
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': classifier_lr,
                    'name': 'classifier'
                }
            ]
            
            print(f"Using differentiated learning rates:")
            print(f"  Encoder LR: {encoder_lr}")
            print(f"  Classifier LR: {classifier_lr}")
            
        else:
            # If encoder is frozen, only optimize classifier parameters
            param_groups = [
                {
                    'params': self.classifier.parameters(),
                    'lr': self.config.learning_rate,
                    'name': 'classifier'
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