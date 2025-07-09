#!/bin/bash
#SBATCH --job-name=diffusion-80gb
#SBATCH --output=diffusion-80gb-%j.out
#SBATCH --error=diffusion-80gb-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00

# Load conda (adjust path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-env

# Print GPU info for debugging
nvidia-smi

# Run the training script
python Reflected-Diffusion/run_train.py \
    data=gto_halo \
    model=ncsnpp \
    --config-dir=Reflected-Diffusion/configs \
    --config-name=train

# After training, plot the loss curve
python plot_losses.py
