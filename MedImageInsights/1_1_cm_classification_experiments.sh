#!/bin/bash

# CM Classification Cross-Validation Runner
# Runs 5-fold cross-validation across multiple GPUs in parallel

# Create output directory if it doesn't exist
mkdir -p ./outs

echo "Starting CM classification cross-validation experiments..."
echo "$(date): Launching 5 parallel processes across GPUs 1-5"

# Launch all 5 folds in parallel
nohup python 1_1_cm_classification_experiments.py --gpu_num 0 --fold 0 > ./outs/cm_classification0.out 2>&1 &
PID1=$!

nohup python 1_1_cm_classification_experiments.py --gpu_num 3 --fold 1 > ./outs/cm_classification1.out 2>&1 &
PID2=$!

nohup python 1_1_cm_classification_experiments.py --gpu_num 5 --fold 2 > ./outs/cm_classification2.out 2>&1 &
PID3=$!

nohup python 1_1_cm_classification_experiments.py --gpu_num 6 --fold 3 > ./outs/cm_classification3.out 2>&1 &
PID4=$!

nohup python 1_1_cm_classification_experiments.py --gpu_num 7 --fold 4 > ./outs/cm_classification4.out 2>&1 &
PID5=$!

# Store PIDs for monitoring
echo "Process IDs:"
echo "Fold 0 (GPU 1): $PID1"
echo "Fold 1 (GPU 2): $PID2"
echo "Fold 2 (GPU 3): $PID3"
echo "Fold 3 (GPU 4): $PID4"
echo "Fold 4 (GPU 5): $PID5"

# Save PIDs to file for later reference
echo "$PID1 $PID2 $PID3 $PID4 $PID5" > ./outs/cm_pids.txt


