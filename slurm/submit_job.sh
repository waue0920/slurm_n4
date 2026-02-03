#!/bin/bash
#SBATCH --job-name=my_multi_gpu_job
#SBATCH --nodes=5
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=my_multi_gpu_job_%j.out
#SBATCH --error=my_multi_gpu_job_%j.err

echo "Starting job on $SLURM_NNODES nodes"
echo "Allocated GPUs: $SLURM_GPUS"

# Placeholder for your actual command
# Example: srun python your_script.py
# Example: srun --mpi=pmi2 your_mpi_application
echo "Please replace this line with your actual job command."
