#!/bin/bash
#SBATCH --job-name=train_swiss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=shared
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=./logs/train_swiss.out
#SBATCH --error=./logs/train_swiss.err


module load gcc/14.1.0 python/3.12.1 cuda/12.3.0

python -u train_swiss.py --k 3 --num_layer 4 --hidden_dim 100 &
python -u train_swiss.py --k 3 --num_layer 4 --hidden_dim 100 --standardize &
python -u train_swiss.py --k 3 --num_layer 6 --hidden_dim 150 --lr 0.0001 --n_epochs 5000 &

# Wait for all background jobs to finish
wait

echo "All training jobs have completed."