#!/work/u00cjz00/binary/bash5.0/bin/bash


#SBATCH -o nemo_continoues_pretrain_%j.out      ### Log folder, Here %j is job ID
#SBATCH -e nemo_continoues_pretrain_%j.err      ### Log folder, Here %j is job ID
#SBATCH --job-name=twcc                         ## job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virginia.chen2007@gmail.com
#SBATCH --nodes=4                              ## Request 4 nodes
#SBATCH --ntasks-per-node=1                     ## 1 srun tasks per node
#SBATCH --cpus-per-task=8                       ## 8 CPUs per srun task
#SBATCH --gres=gpu:4                            ## 8 GPUs per node
#SBATCH --account="GOV113054"                   ## Project ID for accounting
#SBATCH --partition=gp2d                       ## Queue selection2
#SBATCH --mem=90GB     
module load singularity

# Define paths and environment variables
SIF=/work/waue0920/open_access/ngc2407-nemo-20240901.sif
SINGULARITY="singularity run --nv $SIF"

MODEL_NAME=llama3_8b
MODEL=meta-llama/Meta-Llama-3-8B.nemo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TP=8
PP=4
LR=1e-4
DATA_PREFIX="[1.0,data/custom_dataset/preprocessed/wikinews_text_document]"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi

# Prepare the Python run command
PTH_RUN="python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path=/opt/NeMo-Framework-Launcher/launcher_scripts/conf/training/llama --config-name=llama3_8b \
+base_results_dir=results \
+model.seq_len_interpolation_factor=null \
trainer.num_nodes=4 \
trainer.devices=8 \
trainer.precision=16-mixed \
trainer.max_steps=500 \
trainer.limit_val_batches=1 \
trainer.val_check_interval=100 \
trainer.accumulate_grad_batches=1 \
exp_manager.explicit_log_dir=/home/virginia1988/results/$MODEL_NAME/Pretraining \
exp_manager.wandb_logger_kwargs.name=$MODEL_NAME \
exp_manager.create_checkpoint_callback=True \
exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
exp_manager.checkpoint_callback_params.model_parallel_size=$(($TP*$PP)) \
+exp_manager.checkpoint_callback_params.every_n_train_steps=50 \
+exp_manager.checkpoint_callback_params.every_n_epochs=null \
exp_manager.checkpoint_callback_params.monitor=\"epoch\" \
exp_manager.checkpoint_callback_params.save_top_k=-1 \
model.num_query_groups=32 \
model.micro_batch_size=1 \
model.global_batch_size=2 \
model.context_parallel_size=1 \
model.tensor_model_parallel_size=$TP \
model.pipeline_model_parallel_size=$PP \
model.optim.lr=$LR \
model.data.splits_string='9990,8,2' \
model.data.data_prefix=${DATA_PREFIX} \
model.data.num_workers=1 
"

#model.num_layers=22 \
#model.hidden_size=2048 \
#model.ffn_hidden_size=5632 \
#model.num_attention_heads=32 \
#model.kv_channels=128 \

export NCCL_DEBUG=INFO

# Run the job using srun with Singularity
srun --mpi=pmix $SINGULARITY $PTH_RUN
