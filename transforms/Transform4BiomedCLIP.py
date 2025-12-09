import monai as mn
import torch
import numpy as np
from monai.transforms import MapTransform
from monai.config import KeysCollection
from open_clip import create_model_from_pretrained
from PIL import Image

class BiomedCLIPProcessor(MapTransform):
    """
    A MONAI transform that processes images for RAD-DINO (Radiological Dense Image Network Operations).
    
    This transform converts input medical images to the format required by the RAD-DINO model:
    - Ensures images have 3 channels
    - Normalizes pixel values
    - Processes images using the pretrained RAD-DINO processor
    - Returns tensors of consistent size
    
    Args:
        keys (KeysCollection): Keys of the corresponding items to be transformed.
        processor_name (str): The pretrained model name for AutoImageProcessor.
            Default: "microsoft/rad-dino"
        im_size (int): Target image size (both height and width) after processing.
            Default: 518
    """
    def __init__(self, keys: KeysCollection, processor_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', image_size: int = 224):
        super().__init__(keys)
        _, self.processor = create_model_from_pretrained(processor_name)
        
        #self.processor.size = {"shortest_edge": im_size}    
        #self.processor.crop_size = {"height": im_size, "width": im_size}
        
        self.image_size = image_size
        
    def _convert_to_3channel(self, img_array: np.ndarray) -> np.ndarray:
        """
        Convert single-channel images to 3-channel by repeating the data.
        
        Args:
            img_array (np.ndarray): Input image array with shape [H,W] or [1,H,W]
            
        Returns:
            np.ndarray: 3-channel image with shape [3,H,W]
        """
        if len(img_array.shape) == 2:  # Single channel [H,W]
            return np.stack([img_array] * 3, axis=0)  # Makes it [3,H,W]
        elif len(img_array.shape) == 3 and img_array.shape[0] == 1:  # [1,H,W]
            return np.repeat(img_array, 3, axis=0)  # Makes it [3,H,W]
        return img_array
    
    def __call__(self, data):
        """
        Process the input data dictionary.
        
        Args:
            data (dict): Input data dictionary containing images to process
            
        Returns:
            dict: Updated dictionary with processed image tensors
            
        Raises:
            ValueError: If processed tensor doesn't match expected dimensions
        """
        d = dict(data)
        for key in self.keys:
            # Get image array
            img_array = d[key]
            
            # Convert to 3 channels if needed
            img_array = self._convert_to_3channel(img_array)
            
            # Convert to [H,W,C] for PIL
            img_array = np.transpose(img_array, (1, 2, 0))
            
            # Normalize to [0,255] if not already
            if img_array.dtype != np.uint8:
                img_array = ((img_array - img_array.min()) / 
                           (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL
            pil_image = Image.fromarray(img_array)
            
            # Process using RAD-DINO processor
            processed = self.processor(pil_image)

            # Verify tensor dimensions
            if processed.shape != (3, self.image_size, self.image_size):
                raise ValueError(f"Unexpected tensor shape: {processed.shape}")
            
            # Store processed tensor
            d[key] = processed
           
        return d


class Transform4BiomedCLIP:
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
                BiomedCLIPProcessor(keys=["img"], image_size=IMG_SIZE),
    
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
            BiomedCLIPProcessor(keys=["img", "msk"], image_size=IMG_SIZE),
            mn.transforms.ScaleIntensityRangePercentilesd(keys=["msk"], lower=0, upper=100, b_min=0, b_max=1, clip=True, channel_wise=True),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=True, cval=0.0),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=False, cval=1.0),
            mn.transforms.SelectItemsD(keys=["img", "paths", "msk"]),
            mn.transforms.ToTensorD(keys=["img", "msk"], dtype=torch.float, track_meta=False)
        ])


