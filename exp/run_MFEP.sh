#!/bin/bash
#SBATCH --account=leban0
#SBATCH --partition=spgpu
#SBATCH --job-name=MFEP-betatc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --output=./logs/mfep-betatc.out
#SBATCH --error=./logs/mfep-betatc.err


# Load any necessary modules
module load gcc/14.1.0 python/3.12.1 cuda/12.3.0

# Activate your conda environment if needed

python -u MFEP_comparisons.py



