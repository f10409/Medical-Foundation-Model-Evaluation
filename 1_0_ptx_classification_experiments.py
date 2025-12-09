#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import subprocess
import sys
import argparse
from tqdm import tqdm
import os
from utils.fine_tune_classification_model import fine_tune_classification_model
from utils.evaluate_classification_model import evaluate_classification_model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run medical image classification experiments')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use (default: 0)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation (default: 0)')
    
    args = parser.parse_args()
    
    GPU_NUM = args.gpu_num
    FOLD = args.fold
    
    print(f"Using GPU: {GPU_NUM}, Fold: {FOLD}")
    
    # Define experiments
    experiments = [
        {
            'model_name': "nvidia/segformer-b4-finetuned-ade-512-512",
            'freeze_encoder': False,
            'test_name': f'SegFormer_ptx_cla_e2e_{FOLD}',
            'GPU': GPU_NUM
        },
        # {
        #     'model_name': "google/medsiglip-448",
        #     'freeze_encoder': True,
        #     'test_name': f'MedSigLIP_ptx_cla_fz_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "google/medsiglip-448",
        #     'freeze_encoder': False,
        #     'test_name': f'MedSigLIP_ptx_cla_e2e_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli",
        #     'freeze_encoder': True,
        #     'test_name': f'CheXagent_ptx_cla_fz_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli",
        #     'freeze_encoder': False,
        #     'test_name': f'CheXagent_ptx_cla_e2e_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        #     'freeze_encoder': True,
        #     'test_name': f'BiomedCLIP_ptx_cla_fz_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        #     'freeze_encoder': False,
        #     'test_name': f'BiomedCLIP_ptx_cla_e2e_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "microsoft/rad-dino",
        #     'freeze_encoder': True,
        #     'test_name': f'RAD-DINO_ptx_cla_fz_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "microsoft/rad-dino",
        #     'freeze_encoder': False,
        #     'test_name': f'RAD-DINO_ptx_cla_e2e_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "google/siglip2-so400m-patch16-512",
        #     'freeze_encoder': True,
        #     'test_name': f'SigLIP2_ptx_cla_fz_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "google/siglip2-so400m-patch16-512",
        #     'freeze_encoder': False,
        #     'test_name': f'SigLIP2_ptx_cla_e2e_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "facebook/dinov2-base",
        #     'freeze_encoder': True,
        #     'test_name': f'DINOv2_ptx_cla_fz_{FOLD}',
        #     'GPU': GPU_NUM
        # },
        # {
        #     'model_name': "facebook/dinov2-base",
        #     'freeze_encoder': False,
        #     'test_name': f'DINOv2_ptx_cla_e2e_{FOLD}',
        #     'GPU': GPU_NUM
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
from utils.fine_tune_classification_model import fine_tune_classification_model
from utils.evaluate_classification_model import evaluate_classification_model
from config_ptx_cla import config

config.model_name = "{exp['model_name']}"
config.freeze_encoder = {exp['freeze_encoder']}
config.test_name = "{exp['test_name']}"
config.weight_path = "./weights/{exp['test_name']}"
config.GPU = {exp['GPU']}
config.input_path = "./inputs/input_train_ptx_cla_{FOLD}.csv"

fine_tune_classification_model(config=config)
results = evaluate_classification_model(config=config)

# Save results to file
predictions_df = pd.DataFrame({{k:np.squeeze(results['test_predictions'][k]) for k in results['test_predictions'].keys()}})
predictions_df['freeze'] = {int(exp['freeze_encoder'])}
predictions_df['model'] = "{exp['test_name']}".split('_')[0]
predictions_df['task'] = "classification"
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