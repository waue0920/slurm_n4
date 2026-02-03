#!/work/u00cjz00/binary/bash5.0/bin/bash

#SBATCH -o nemo_download_dataset_%j.out        # Log file for output, %j is job ID
#SBATCH -e nemo_download_dataset_%j.err        # Log file for errors, %j is job ID
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

# Install the `datasets` package within the Singularity container
INSTALL_PACKAGES="
pip install datasets
"

# Python script to download the dataset
DATA_DOWNLOAD="
from datasets import load_dataset
dataset = load_dataset('erhwenkuo/wikinews-zhtw')['train']
dataset.to_json('./data/custom_dataset/json/wikinews-zhtw.jsonl', force_ascii=False)
print('finished')
"

# Create necessary directories
mkdir -p data/custom_dataset/json
mkdir -p data/custom_dataset/preprocessed

# Install required packages
srun --mpi=pmix $SINGULARITY bash -c "$INSTALL_PACKAGES"

# Download the dataset
srun --mpi=pmix $SINGULARITY python -c "$DATA_DOWNLOAD"

# Prepare the Python run command
PTH_RUN="
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \\
--input=data/custom_dataset/json/wikinews-zhtw.jsonl \\
--json-keys=text \\
--dataset-impl mmap \\
--tokenizer-library=huggingface \\
--tokenizer-type TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
--output-prefix=data/custom_dataset/preprocessed/wikinews \\
--append-eod
"

export NCCL_DEBUG=INFO

# Run the preprocessing job using srun with Singularity
srun --mpi=pmix $SINGULARITY bash -c "$PTH_RUN"
