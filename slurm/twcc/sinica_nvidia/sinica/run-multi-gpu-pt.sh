#!/bin/bash
export MODEL_NAME=TinyLlama-1.1B
export MODEL="/models/24.05_hf2nemo_models/TinyLlama/TinyLlama-1.1B-Chat-v1.0.nemo"
# export NUM_GPUS=$(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_NODE))
export NUM_GPUS=2
export tensor_model_parallel_size=1
export pipeline_model_parallel_size=2
export MAX_STEPS=100
export per_device_train_batch_size=1
export global_batch_size=4
export learning_rate=5e-6
export DATA_SPLITS=\'9990,8,2\'
export DATA_PREFIX=[1.0,/data/custom_dataset/preprocessed/wikinews_text_document]
export HYDRA_FULL_ERROR=1

export save_steps=50

export WANDB_PROJECT='twcc-slurm-nemo_llama_pretrain'
export NUM_NODES=${SLURM_JOB_NUM_NODES}

output_dir="outputs/job-${SLURM_JOB_ID}-results/"
current_script=$0

mkdir -p ${output_dir}
cp ${current_script} ${output_dir}

python modified_24.05_megatron_gpt_continue_training.py \
    --config-path=/opt/NeMo-Framework-Launcher/launcher_scripts/conf/training/llama --config-name=llama2_7b  \
    +restore_from_path=$MODEL \
    +base_results_dir=${output_dir} \
    +model.seq_len_interpolation_factor=null \
    trainer.num_nodes=${NUM_NODES} \
    trainer.devices=$NUM_GPUS \
    trainer.precision=16 \
    trainer.max_steps=$MAX_STEPS \
    trainer.limit_val_batches=32 \
    trainer.val_check_interval=100 \
    trainer.log_every_n_steps=1 \
    hydra.job.chdir=True \
    exp_manager.explicit_log_dir=${output_dir}/$MODEL_NAME/Pretraining \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name=${SLURM_JOB_ID}-${MODEL_NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    exp_manager.checkpoint_callback_params.model_parallel_size=$(($tensor_model_parallel_size*$pipeline_model_parallel_size)) \
    +exp_manager.checkpoint_callback_params.every_n_train_steps=${save_steps} \
    +exp_manager.checkpoint_callback_params.every_n_epochs=null \
    exp_manager.checkpoint_callback_params.monitor="epoch" \
    exp_manager.checkpoint_callback_params.save_top_k=-1 \
    model.micro_batch_size=$per_device_train_batch_size \
    model.global_batch_size=$global_batch_size \
    model.tensor_model_parallel_size=$tensor_model_parallel_size \
    model.pipeline_model_parallel_size=$pipeline_model_parallel_size \
    model.tokenizer.library=huggingface \
    model.tokenizer.type=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    model.tokenizer.model=null \
    model.optim.lr=$learning_rate \
    model.data.splits_string=${DATA_SPLITS} \
    model.data.data_prefix=${DATA_PREFIX} \
    model.data.num_workers=0 \
    model.data.seq_length=1024

