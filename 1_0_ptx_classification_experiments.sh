#!/bin/bash

# PTX Classification Cross-Validation Runner
# Runs 5-fold cross-validation across multiple GPUs in parallel

# Create output directory if it doesn't exist
mkdir -p ./outs

echo "Starting PTX classification cross-validation experiments..."
echo "$(date): Launching 5 parallel processes across GPUs 1-5"

# Launch all 5 folds in parallel
nohup python 1_0_ptx_classification_experiments.py --gpu_num 0 --fold 0 > ./outs/ptx_classification0.out 2>&1 &
PID1=$!

nohup python 1_0_ptx_classification_experiments.py --gpu_num 2 --fold 1 > ./outs/ptx_classification1.out 2>&1 &
PID2=$!

nohup python 1_0_ptx_classification_experiments.py --gpu_num 3 --fold 2 > ./outs/ptx_classification2.out 2>&1 &
PID3=$!

nohup python 1_0_ptx_classification_experiments.py --gpu_num 4 --fold 3 > ./outs/ptx_classification3.out 2>&1 &
PID4=$!

nohup python 1_0_ptx_classification_experiments.py --gpu_num 5 --fold 4 > ./outs/ptx_classification4.out 2>&1 &
PID5=$!

# Store PIDs for monitoring
echo "Process IDs:"
echo "Fold 0 (GPU 1): $PID1"
echo "Fold 1 (GPU 2): $PID2"
echo "Fold 2 (GPU 3): $PID3"
echo "Fold 3 (GPU 4): $PID4"
echo "Fold 4 (GPU 5): $PID5"

# Save PIDs to file for later reference
echo "$PID1 $PID2 $PID3 $PID4 $PID5" > ./outs/ptx_pids.txt

