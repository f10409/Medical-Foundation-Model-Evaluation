import monai as mn
import torch
import numpy as np
from monai.transforms import MapTransform
from monai.config import KeysCollection
from PIL import Image
import torchvision.transforms as T

class MedImageInsightsProcessor(MapTransform):

    def __init__(self, keys: KeysCollection, image_size: int = 480):
        super().__init__(keys)
                
        self.image_size = image_size
        
    def _convert_to_3channel(self, img_array: np.ndarray) -> np.ndarray:

        if len(img_array.shape) == 2:  # Single channel [H,W]
            return np.stack([img_array] * 3, axis=0)  # Makes it [3,H,W]
        elif len(img_array.shape) == 3 and img_array.shape[0] == 1:  # [1,H,W]
            return np.repeat(img_array, 3, axis=0)  # Makes it [3,H,W]
        return img_array


    def unicl_eval_preprocessing(self,        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        interpolation='bicubic',
        center_crop=False):
    
        # Interpolation mapping
        INTERPOLATION_MODES = {
            'bilinear': T.InterpolationMode.BILINEAR,
            'bicubic': T.InterpolationMode.BICUBIC,
            'nearest': T.InterpolationMode.NEAREST,
        }
        
        normalize = T.Normalize(mean=mean, std=std)
        interpolation_mode = INTERPOLATION_MODES[interpolation]
        
        if center_crop:
            # Resize larger then center crop
            resize_size = int(self.image_size / 0.875)
            transforms = T.Compose([
                T.Resize(resize_size, interpolation=interpolation_mode),
                T.CenterCrop(slef.image_size),
                T.ToTensor(),
                normalize,
            ])
        else:
            # Direct resize to target size
            transforms = T.Compose([
                T.Resize((self.image_size,self.image_size), interpolation=interpolation_mode),
                T.ToTensor(),
                normalize,
            ])
        
        return transforms
    
    def __call__(self, data):

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
            
            # Process using processor
            transform = self.unicl_eval_preprocessing()
            processed = transform(pil_image)

            # Verify tensor dimensions
            if processed.shape != (3, self.image_size, self.image_size):
                raise ValueError(f"Unexpected tensor shape: {processed.shape}")
            
            # Store processed tensor
            d[key] = processed
           
        return d


class Transform4MedImageInsights:

    def __init__(self, CLASSES=None):

        if CLASSES is not None:
            self.cla_transforms = mn.transforms.Compose([
                mn.transforms.LoadImageD(keys="img", reader="PILReader", ensure_channel_first=True),
                mn.transforms.Transposed(keys=["img"], indices=[0, 2, 1]),
                MedImageInsightsProcessor(keys=["img"]),
    
                mn.transforms.ToTensorD(keys=[*CLASSES], dtype=torch.float),            
                mn.transforms.ConcatItemsD(keys=[*CLASSES], name='cls'),
                
                mn.transforms.SelectItemsD(keys=["img", "paths", "cls"]),
                mn.transforms.ToTensorD(keys=["img"], dtype=torch.float, track_meta=False),
                mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float)])
        else:
            self.cla_transforms = None

        self.seg_transforms = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys=["img", "msk"], reader="PILReader", ensure_channel_first=True),
            mn.transforms.Transposed(keys=["img", "msk"], indices=[0, 2, 1]),
            MedImageInsightsProcessor(keys=["img", "msk"]),
            mn.transforms.ScaleIntensityRangePercentilesd(keys=["msk"], lower=0, upper=100, b_min=0, b_max=1, clip=True, channel_wise=True),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=True, cval=0.0),
            mn.transforms.ThresholdIntensityd(keys=["msk"], threshold=0.66, above=False, cval=1.0),
            mn.transforms.SelectItemsD(keys=["img", "paths", "msk"]),
            mn.transforms.ToTensorD(keys=["img", "msk"], dtype=torch.float, track_meta=False)
        ])