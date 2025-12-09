import monai as mn
import torch
import numpy as np
from monai.transforms import MapTransform
from transformers import AutoImageProcessor
from PIL import Image


class MedSigLIPProcessor(MapTransform):
    """
    MONAI transform for processing medical images using Google's MedSigLIP processor.
    
    Converts medical images to 8-bit RGB format and applies MedSigLIP preprocessing.

    Args:
        keys (KeysCollection): Keys of the images to be transformed
        processor_name (str): HuggingFace model repository name (default: "google/medsiglip-448")
        max_length (int, optional): Maximum sequence length for processor compatibility (default: None)
    """
    
    def __init__(self, keys, processor_name="google/medsiglip-448", max_length=None):
        super().__init__(keys)
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.max_length = max_length
          
    def convert_to_8bit_3channel(self, img_array):
        """
        Convert medical image array to 8-bit 3-channel RGB format.
        
        Args:
            img_array (np.ndarray): Input image array of shape (H, W) or (H, W, C)
            
        Returns:
            PIL.Image: RGB PIL Image in 8-bit format
        """
        # Convert to float for processing
        img_array = img_array.astype(float)   
        
        # Handle different input dimensions
        if img_array.ndim == 2:  # Grayscale
            # Normalize to 0-255 range preserving dynamic range
            min_val = img_array.min()
            max_val = img_array.max()
            
            if max_val > min_val:  # Avoid division by zero
                normalized = ((img_array - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
            else:
                normalized = np.zeros_like(img_array, dtype=np.uint8)
            
            # Convert to 3-channel by repeating grayscale values
            rgb_array = np.stack([normalized, normalized, normalized], axis=-1)
            
        elif img_array.ndim == 3:  # Multi-channel

            if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
                img_array = np.transpose(img_array, (1, 2, 0))
            
            # Process each channel independently to preserve dynamic range
            channels = []
            for i in range(img_array.shape[2]):
                channel = img_array[:, :, i]
                min_val = channel.min()
                max_val = channel.max()
                
                if max_val > min_val:
                    normalized_channel = ((channel - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
                else:
                    normalized_channel = np.zeros_like(channel, dtype=np.uint8)
                
                channels.append(normalized_channel)
            
            # If input has more or fewer than 3 channels, adjust accordingly
            if len(channels) == 1:  # Single channel
                rgb_array = np.stack([channels[0], channels[0], channels[0]], axis=-1)
            elif len(channels) >= 3:  # Three or more channels - take first 3
                rgb_array = np.stack(channels[:3], axis=-1)
            else:  # Two channels - duplicate first channel
                rgb_array = np.stack([channels[0], channels[1], channels[0]], axis=-1)
        
        else:
            raise ValueError(f"Unsupported image dimensions: {img_array.ndim}. Expected 2D or 3D array.")
        
        # Convert back to PIL Image in RGB mode
        return Image.fromarray(rgb_array, mode='RGB')
    
    def __call__(self, data):
        """
        Applies the MedSigLIP transformation pipeline to the input data.
        
        Args:
            data (dict): Input data dictionary containing image data
            
        Returns:
            dict: Processed data with normalized tensor images
        """
        d = dict(data)
        
        for key in self.keys:
            if key not in d:
                raise KeyError(f"Key '{key}' not found in input data. Available keys: {list(d.keys())}")
                
            try:
                # Convert image array to proper format
                img_array = d[key]
                
                image = self.convert_to_8bit_3channel(img_array)          
                
                # Prepare processor arguments for image preprocessing
                processor_kwargs = {
                    "images": image,
                    "return_tensors": "pt"    # Return PyTorch tensors
                }
                
                # Add max_length if specified (for compatibility with some processors)
                if self.max_length is not None:
                    processor_kwargs["max_length"] = self.max_length
                    
                # Process the image through MedSigLIP processor
                inputs = self.processor(**processor_kwargs)
                
                # Remove batch dimension added by processor (DataLoader will add it back)
                processed_data = {
                    k: tensor.squeeze(0) for k, tensor in inputs.items()
                }

                # Extract the processed pixel values
                processed = processed_data['pixel_values']
                
                # Update the data dictionary with processed image
                d[key] = processed
                
            except Exception as e:
                raise RuntimeError(f"Failed to process image for key '{key}': {str(e)}") from e
           
        return d


class Transform4MedSigLIP:
    """
    MONAI transform composition for processing medical images with MedSigLIP.
    
    Complete preprocessing pipeline for medical image analysis tasks.
    
    Args:
        IMG_SIZE (int): Target spatial size for input images
        CLASSES (list): List of class names for classification tasks
    """
    
    def __init__(self, CLASSES=None):
        """Initialize the transformation pipeline."""
        
        if CLASSES is not None:
            self.cla_transforms = mn.transforms.Compose([
                # Load images with ITK reader ensuring channel-first format
                mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
                
                # Transpose axes to correct orientation (swap height and width)
                mn.transforms.Transposed(keys=["img"], indices=[0, 2, 1]),
                
                # Apply MedSigLIP preprocessing
                MedSigLIPProcessor(keys=["img"]),
    
                # Convert classification labels to tensors
                mn.transforms.ToTensorD(keys=[*CLASSES], dtype=torch.float),            
                
                # Concatenate all class labels into a single tensor
                mn.transforms.ConcatItemsD(keys=[*CLASSES], name='cls'),
                
                # Select only the required keys for model input
                mn.transforms.SelectItemsD(keys=["img", "paths", "cls"]),
                
                # Final tensor conversions
                mn.transforms.ToTensorD(keys=["img"], dtype=torch.float, track_meta=False),
                mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float)
            ])
        else:
            self.cla_transforms = None

        self.seg_transforms = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys=["img", "msk"], reader="ITKReader", ensure_channel_first=True),
            mn.transforms.Transposed(keys=["img", "msk"], indices=[0, 2, 1]),
            MedSigLIPProcessor(keys=["img", "msk"]),
            mn.transforms.ScaleIntensityRangePercentilesd(keys=["msk"], lower=0, upper=100, b_min=0, b_max=1, clip=True, channel_wise=True),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=True, cval=0.0),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=False, cval=1.0),
            mn.transforms.SelectItemsD(keys=["img", "paths", "msk"]),
            mn.transforms.ToTensorD(keys=["img", "msk"], dtype=torch.float, track_meta=False)
        ])