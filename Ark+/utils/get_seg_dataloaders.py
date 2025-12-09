from torch.utils.data import DataLoader
from transforms.Transform4Ark import Transform4Ark
# from transforms.Transform4BiomedCLIP import Transform4BiomedCLIP
# from transforms.Transform4RADDINO import Transform4RADDINO
# from transforms.Transform4CheXagent import Transform4CheXagent
# from transforms.Transform4MedSigLIP import Transform4MedSigLIP
# from transforms.Transform4SigLIP2 import Transform4SigLIP2
# from transforms.Transform4DINOv2 import Transform4DINOv2
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import itertools
from tqdm import tqdm
import monai as mn


def get_data_dict_part(df_part, config):

    BASE_PATH = config.img_base_path
    MASK_PATH = config.msk_base_path
    IMG_PATH_COLUMN_NAME = config.IMG_PATH_COLUMN_NAME
    
    data_dict = list()
    for i in tqdm(range(len(df_part)), desc="Processing part"):
        row = df_part.iloc[i]

        data_dict.append({
            'img':f'{BASE_PATH}/'+row[f"{IMG_PATH_COLUMN_NAME}"],
            "paths": f'{BASE_PATH}/'+row[f"{IMG_PATH_COLUMN_NAME}"],
            'msk':f'{MASK_PATH}/'+row[f"{IMG_PATH_COLUMN_NAME}"][:-4]+f'_{config.mask_suffix}.png',
        })
    
    return data_dict

def get_data_dict(df, config, num_cores=8):
    # Split dataframe into parts
    parts = np.array_split(df, num_cores)
    
    # Create partial function with config argument
    func = partial(get_data_dict_part, config=config)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        data_dicts = executor.map(func, parts)
    
    # Flatten the list of lists
    return list(itertools.chain(*data_dicts))


def get_seg_dataloader(input_df, config):
    """
    Create train, validation, and test dataloaders from input dataframe.
    
    Args:
        input_df: DataFrame containing image paths, labels, and split information
        config: Configuration object containing img_base_path, labels, and batch_size
    
    Returns:
        tuple: (train_dataloader, valid_dataloader, test_dataloader)
    """
    # Prepare training data
    train_df = input_df[input_df.Split == 'Train'].reset_index(drop=True)#.iloc[0:1000]
    
    # Prepare validation data
    val_df = input_df[input_df.Split == 'Valid'].reset_index(drop=True)#.iloc[0:100]
    
    # Prepare test data
    test_df = input_df[input_df.Split == 'Test'].reset_index(drop=True)
    
    train_dict = get_data_dict(train_df, config)
    val_dict = get_data_dict(val_df, config)
    test_dict = get_data_dict(test_df, config)

    # if config.model_name == "google/medsiglip-448":
    #     train_transforms = Transform4MedSigLIP(CLASSES=None).seg_transforms
    #     val_transforms = Transform4MedSigLIP(CLASSES=None).seg_transforms
    # if config.model_name == "microsoft/rad-dino":
    #     train_transforms = Transform4RADDINO(IMG_SIZE=518, CLASSES=None).seg_transforms
    #     val_transforms = Transform4RADDINO(IMG_SIZE=518, CLASSES=None).seg_transforms
    # if config.model_name == "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli":    
    #     train_transforms = Transform4CheXagent(IMG_SIZE=512, CLASSES=None).seg_transforms
    #     val_transforms = Transform4CheXagent(IMG_SIZE=512, CLASSES=None).seg_transforms
    # if config.model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 
    #     train_transforms = Transform4BiomedCLIP(IMG_SIZE=224, CLASSES=None).seg_transforms
    #     val_transforms = Transform4BiomedCLIP(IMG_SIZE=224, CLASSES=None).seg_transforms
    # if config.model_name == "google/siglip2-so400m-patch16-512":
    #     train_transforms = Transform4SigLIP2(CLASSES=None).seg_transforms
    #     val_transforms = Transform4SigLIP2(CLASSES=None).seg_transforms
    # if config.model_name == "facebook/dinov2-base":
    #     train_transforms = Transform4DINOv2(CLASSES=None).seg_transforms
    #     val_transforms = Transform4DINOv2(CLASSES=None).seg_transforms
    # if config.model_name == "MedImageInsights":
    #     train_transforms = Transform4MedImageInsights(CLASSES=None).seg_transforms
    #     val_transforms = Transform4MedImageInsights(CLASSES=None).seg_transforms

    if config.model_name == "Ark+":
        train_transforms = Transform4Ark(CLASSES=None).seg_transforms
        val_transforms = Transform4Ark(CLASSES=None).seg_transforms

    # define datasets
    train_ds = mn.data.Dataset(data=train_dict, transform=train_transforms)
    val_ds = mn.data.Dataset(data=val_dict, transform=val_transforms)
    test_ds = mn.data.Dataset(data=test_dict, transform=val_transforms)
    
    # define data loader
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, persistent_workers=True, num_workers=config.num_workers, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=False, persistent_workers=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=False, persistent_workers=True, pin_memory=True)
    
    return train_dl, val_dl, test_dl