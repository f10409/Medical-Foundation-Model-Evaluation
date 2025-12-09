import monai as mn
import torch
import numpy as np
from monai.transforms import MapTransform
from transformers import AutoImageProcessor
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class CheXagentProcessor(MapTransform):
    """
    MONAI transform for processing medical images using Stanford AIMI's XraySigLIP processor.
    Handles image resizing, normalization, and conversion to tensor format.

    Args:
        keys (KeysCollection): Keys of the images to be transformed
        processor_name (str): HuggingFace model repository name (default: "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli")
        image_size (int): Target size for image resizing (default: 512)
    """
    def __init__(self, keys, processor_name="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", image_size=512):
        super().__init__(keys)
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.image_size = image_size
        
        self.mean = self.processor.image_mean
        self.std = self.processor.image_std
        
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        
    def _convert_to_3channel(self, img_array):
        """
        Converts grayscale images to RGB by repeating the channel.
        
        Args:
            img_array (np.ndarray): Input image array in [H,W] or [1,H,W] format
            
        Returns:
            np.ndarray: RGB image array in [3,H,W] format
        """
        if len(img_array.shape) == 2:  # Single channel [H,W]
            return np.stack([img_array] * 3, axis=0)
        elif len(img_array.shape) == 3 and img_array.shape[0] == 1:  # [1,H,W]
            return np.repeat(img_array, 3, axis=0)
        return img_array
    
    def _array_to_pil(self, img_array):
        """
        Converts numpy array to PIL Image with appropriate channel ordering.
        
        Args:
            img_array (np.ndarray): Input image array in [C,H,W] or [H,W,C] format
            
        Returns:
            PIL.Image: Converted PIL Image
        """
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
            img_array = np.transpose(img_array, (1, 2, 0))
            
        return Image.fromarray(img_array)
    
    def __call__(self, data):
        """
        Applies the transformation pipeline to the input data.
        
        Args:
            data (dict): Input data dictionary containing image data
            
        Returns:
            dict: Processed data with normalized tensor images
            
        Raises:
            KeyError: If required image key is not found
            ValueError: If image has no contrast or unexpected dimensions
            RuntimeError: If image transformation fails
        """
        d = dict(data)
        for key in self.keys:
            if key not in d:
                raise KeyError(f"Key {key} not found in input data")
                
            img_array = d[key]
            img_array = self._convert_to_3channel(img_array)          
            
            # Normalize to [0,255] range
            if img_array.dtype != np.uint8:
                min_val = img_array.min()
                max_val = img_array.max()
                try:
                    if max_val == min_val:
                        raise ValueError(f"Image has no contrast: min={min_val}, max={max_val}")
                    
                    img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                except:
                    img_array = np.zeros(img_array.shape).astype(np.uint8)
            try:
                pil_image = self._array_to_pil(img_array)
                processed = self.image_transform(pil_image)
            except Exception as e:
                raise RuntimeError(f"Image transformation failed: {str(e)}")
            
            if processed.shape != (3, self.image_size, self.image_size):
                raise ValueError(f"Unexpected tensor shape: {processed.shape}")
            
            d[key] = processed
           
        return d


class Transform4CheXagent:
    """
    Composition of MONAI transforms for processing medical images for the CheXagent model.
    
    Args:
        IMG_SIZE (int): Target spatial size for input images
    """
    def __init__(self, IMG_SIZE, CLASSES=None):
        
        if CLASSES is not None:
            self.cla_transforms = mn.transforms.Compose([
                mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
                mn.transforms.Transposed(keys=["img"], indices=[0, 2, 1]),
                CheXagentProcessor(keys=["img"], image_size=IMG_SIZE),
    
                mn.transforms.ToTensorD(keys=[*CLASSES], dtype=torch.float),            
                mn.transforms.ConcatItemsD(keys=[*CLASSES], name='cls'),
                
                mn.transforms.SelectItemsD(keys=["img", "paths", "cls"]),
                mn.transforms.ToTensorD(keys=["img"], dtype=torch.float, track_meta=False),
                mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float)])
        else:
            self.cla_transforms = None

        self.seg_transforms = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys=["img", "msk"], reader="ITKReader", ensure_channel_first=True),
            mn.transforms.Transposed(keys=["img", "msk"], indices=[0, 2, 1]),
            CheXagentProcessor(keys=["img", "msk"], image_size=IMG_SIZE),
            mn.transforms.ScaleIntensityRangePercentilesd(keys=["msk"], lower=0, upper=100, b_min=0, b_max=1, clip=True, channel_wise=True),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=True, cval=0.0),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=False, cval=1.0),
            mn.transforms.SelectItemsD(keys=["img", "paths", "msk"]),
            mn.transforms.ToTensorD(keys=["img", "msk"], dtype=torch.float, track_meta=False)
        ])
            