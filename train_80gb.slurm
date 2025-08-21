#!/bin/bash
#SBATCH --job-name=gto-1d-diffusion   # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=4            # cpu-cores per task (set to 4 for 4 DataLoader workers)
#SBATCH --mem-per-cpu=4G             # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1                 # number of gpus per node
#SBATCH --constraint=gpu80            # use 80GB GPUs
#SBATCH --time=48:00:00              # total run time limit (HH:MM:SS)
#SBATCH --output=logs/gto-1d-diffusion-%j.out  # standard output log file
#SBATCH --error=logs/gto-1d-diffusion-%j.err   # standard error log file
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

# Run the 1D GTO Halo training with optimized configuration
cd Reflected-Diffusion && python run_train.py --config-name train_gto_1d

