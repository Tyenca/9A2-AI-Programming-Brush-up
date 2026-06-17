#!/bin/bash
#SBATCH --job-name=base_rf
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=outputs/%x-%j.out

# Environment setup (normal shell commands, no #SBATCH)
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/bin/activate

# Go to your repo (adjust if path differs)
cd ~/9A2-AI-Programming-Brush-up/src

python base_rf.py