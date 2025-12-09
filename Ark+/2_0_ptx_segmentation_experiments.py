#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import subprocess
import sys
import argparse
from tqdm import tqdm
import os
from utils.fine_tune_segmentation_model import fine_tune_segmentation_model
from utils.evaluate_segmentation_model import evaluate_segmentation_model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run pneumothorax segmentation experiments')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use (default: 0)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation (default: 0)')
    
    args = parser.parse_args()
    
    GPU_NUM = args.gpu_num
    FOLD = args.fold
    
    print(f"Using GPU: {GPU_NUM}, Fold: {FOLD}")
    
    # Define experiments
    experiments = [
        {
            'model_name': "Ark+",
            'freeze_encoder': True,
            'test_name': f'Ark+(32)_ptx_seg_fz_{FOLD}',
            'GPU': GPU_NUM,
            'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
            'patch_size': 32, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
            'n_channel': 1536  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        },
        {
            'model_name': "Ark+",
            'freeze_encoder': False,
            'test_name': f'Ark+(32)_ptx_seg_e2e_{FOLD}',
            'GPU': GPU_NUM,
            'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
            'patch_size': 32, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
            'n_channel': 1536  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        },
        {
            'model_name': "Ark+",
            'freeze_encoder': True,
            'test_name': f'Ark+(16)_ptx_seg_fz_{FOLD}',
            'GPU': GPU_NUM,
            'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
            'patch_size': 16, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
            'n_channel': 768  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        },
        {
            'model_name': "Ark+",
            'freeze_encoder': False,
            'test_name': f'Ark+(16)_ptx_seg_e2e_{FOLD}',
            'GPU': GPU_NUM,
            'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
            'patch_size': 16, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
            'n_channel': 768  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        },
        # {
        #     'model_name': "Ark+",
        #     'freeze_encoder': True,
        #     'test_name': f'Ark+(8)_ptx_seg_fz_{FOLD}',
        #     'GPU': GPU_NUM,
        #     'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
        #     'patch_size': 8, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
        #     'n_channel': 384  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        # },
        # {
        #     'model_name': "Ark+",
        #     'freeze_encoder': False,
        #     'test_name': f'Ark+(8)_ptx_seg_e2e_{FOLD}',
        #     'GPU': GPU_NUM,
        #     'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
        #     'patch_size': 8, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
        #     'n_channel': 384  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        # },
        # {
        #     'model_name': "Ark+",
        #     'freeze_encoder': True,
        #     'test_name': f'Ark+(4)_ptx_seg_fz_{FOLD}',
        #     'GPU': GPU_NUM,
        #     'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
        #     'patch_size': 4, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
        #     'n_channel': 192  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        # },
        # {
        #     'model_name': "Ark+",
        #     'freeze_encoder': False,
        #     'test_name': f'Ark+(4)_ptx_seg_e2e_{FOLD}',
        #     'GPU': GPU_NUM,
        #     'img_size': 768,  #raddino:518, medsiglip:448, siglip2:512, chexagent:512, biomedclip:224
        #     'patch_size': 4, #raddino:14, medsiglip:14, siglip2:16, chexagent:16, biomedclip:16
        #     'n_channel': 192  #raddino:768, medsiglip:1152, siglip2:1152, chexagent:1024, biomedclip:512/768
        # }
    ]
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Run experiments
    for exp in tqdm(experiments, desc="Running experiments"):
        print(f"Running experiment: {exp['test_name']}")
        
        # Create a separate script for each experiment
        script_content = f"""
import sys
import pandas as pd
import numpy as np
sys.path.append('.')
from utils.fine_tune_segmentation_model import fine_tune_segmentation_model
from utils.evaluate_segmentation_model import evaluate_segmentation_model
from config_ptx_seg import config

config.model_name = "{exp['model_name']}"
config.freeze_encoder = {exp['freeze_encoder']}
config.test_name = "{exp['test_name']}"
config.weight_path = "./weights/{exp['test_name']}"
config.GPU = {exp['GPU']}
config.img_size = {exp['img_size']}
config.patch_size = {exp['patch_size']}
config.n_channel = {exp['n_channel']}
config.input_path = "./inputs/input_train_ptx_seg_{FOLD}.csv"
config.output_folder = f"/mnt/NAS3/projects/fli40/FM_evaluation/pred_masks/{exp['test_name']}"

fine_tune_segmentation_model(config=config)
results = evaluate_segmentation_model(config=config)

# Save results to file
predictions_df = pd.DataFrame(results['test_predictions'])
predictions_df['freeze'] = {int(exp['freeze_encoder'])}
predictions_df['model'] = "{exp['test_name']}".split('_')[0]
predictions_df['task'] = "segmentation"
predictions_df['target'] = "pneumothorax"
predictions_df.to_csv("./results/{exp['test_name']}.csv", index=False)
print("Experiment {exp['test_name']} completed successfully!")
"""
        
        # Write temporary script
        temp_script_name = f'temp_exp_{exp["test_name"]}.py'
        with open(temp_script_name, 'w') as f:
            f.write(script_content)
        
        try:
            # Run the experiment
            result = subprocess.run([sys.executable, temp_script_name], check=True)
            print(f"✓ {exp['test_name']} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"✗ {exp['test_name']} failed with error:")
            print(f"Return code: {e.returncode}")
            print(f"Error output: {e.stderr}")
            if e.stdout:
                print(f"Standard output: {e.stdout}")
        finally:
            # Clean up temporary script
            if os.path.exists(temp_script_name):
                os.remove(temp_script_name)
    
    print("All experiments completed!")

if __name__ == "__main__":
    main()