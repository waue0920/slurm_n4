#!/work/u00cjz00/binary/bash5.0/bin/bash

#SBATCH -o nemo_model_checkpoint_%j.out        # Log file for output, %j is job ID
#SBATCH -e nemo_model_checkpoint_%j.err        # Log file for errors, %j is job ID
#SBATCH --job-name=twcc                        # Job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virginia.chen2007@gmail.com
#SBATCH --nodes=1                              # Request 1 node
#SBATCH --ntasks-per-node=1                    # 1 task per node
#SBATCH --cpus-per-task=2                      # 2 CPUs per task
#SBATCH --gres=gpu:1                           # 1 GPU per node
#SBATCH --account="GOV113054"                  # Project ID for accounting
#SBATCH --partition=gtest                      # Queue selection


module load singularity

# Define paths and environment variables
SIF=/work/waue0920/open_access/ngc2407-nemo-20240901.sif
SINGULARITY="singularity exec --nv $SIF"

# Set the Hugging Face model name as an environment variable
export HF_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Prepare the Python run command
PTH_RUN="
python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \\
--input_name_or_path \$HF_MODEL \\
--output_path TinyLlama-1.1B-Chat-v1.0/TinyLlama-1.1B-Chat-v1.0.nemo
"

# Enable NCCL debugging information
export NCCL_DEBUG=INFO

# Run the conversion job using srun with Singularity
srun --mpi=pmix $SINGULARITY bash -c "$PTH_RUN"
