#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=test-torch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0
module load CUDA/12.9.1
source ~/torch_venv/bin/activate

srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"