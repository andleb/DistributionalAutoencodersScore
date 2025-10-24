#!/bin/bash
#SBATCH --job-name=train_scurvebeta2-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=shared
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=./logs/train_scurvebeta3.out
#SBATCH --error=./logs/train_scurvebeta3.err


module load gcc/14.1.0 python/3.12.1 cuda/12.3.0

python -u train_scurve.py --k 3 --num_layer 4 --hidden_dim 100 --standardize &
python -u train_scurve.py --k 3 --num_layer 6 --hidden_dim 150 --lr 0.0001 --n_epochs 5000 &
python -u train_scurve.py --k 3 --num_layer 6 --hidden_dim 150 --lr 0.0001 --n_epochs 5000 --standardize &

# Wait for all background jobs to finish
wait

echo "All training jobs have completed."