#!/bin/bash
#SBATCH --job-name=train_all_simple
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=./logs/train_all_simple.out
#SBATCH --error=./logs/train_all_simple.err


module load gcc/14.1.0 python/3.12.1 cuda/12.3.0


# List of datasets to train on
DATASETS=("parabola" "exponential" "helix_slice" "grid_sum")

# Loop through the datasets and start training for each in the background
for dataset in "${DATASETS[@]}"; do
    echo "Starting training for dataset: $dataset"
    python -u train_simple.py --dataset "$dataset" &
done

# Wait for all background jobs to finish
wait

echo "All training jobs have completed."