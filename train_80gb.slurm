#!/bin/bash
#SBATCH --job-name=diffusion-40gb    # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=4            # cpu-cores per task (set to 4 for 4 DataLoader workers)
#SBATCH --mem-per-cpu=4G             # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1                 # number of gpus per node
#SBATCH --constraint="amd&gpu40"     # use 40GB GPUs with 100% compute power
#SBATCH --time=24:00:00              # total run time limit (HH:MM:SS)
#SBATCH --output=logs/diffusion-40gb-%j.out  # standard output log file
#SBATCH --error=logs/diffusion-40gb-%j.err   # standard error log file
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-user=se7159@princeton.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda (adjust path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-env

# Print GPU info for debugging
nvidia-smi

# Run the training script
python Reflected-Diffusion/run_train.py --config-dir=Reflected-Diffusion/configs --config-name=train hydra.run.dir="Training Runs/2025.07.14_173150" checkpoint_path="Training Runs/2025.07.14_173150/checkpoints/checkpoint_4.pth"

# After training, plot the loss curve
python plot_losses.py
