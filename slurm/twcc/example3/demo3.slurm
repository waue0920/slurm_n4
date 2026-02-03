#!/bin/bash
#SBATCH -A  GOV109134                                                   ### project number, Example MST109178
#SBATCH -J ds_ex3_n2_g2                                                     ### Job name, Exmaple jupyterlab
#SBATCH -p gp4d                                                         ### Partition Name, Example ngs1gpu
#SBATCH --nodes=2                                                       ### Nodes, Default 1, node number
#SBATCH --ntasks-per-node=1                                             ### Tasks, Default 1, per node tasks
#SBATCH --cpus-per-task=8                                                            ### Cores assigned to each task, Example 4
#SBATCH --gpus-per-node=2                                                    ### GPU number, Example gpu:1
#SBATCH --time=0-1:00:00                                                ### Runnung time, days-hours:minutes:seconds or hours:minutes:seconds
#SBATCH -o demo3.out                                                     ### Log folder, Here %j is job ID
#SBATCH -e demo3.err 


# 環境變數
## GPU數量
export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader |wc -l)
## 主要機器HOSTNAME or IP
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
## 主要機器PORT
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); addr = s.getsockname(); s.close(); print(addr[1])")
#export MASTER_PORT=$(shuf -i 60000-65530 -n 1)
## NCCL 傳輸介面更動, 預設為ib (ib1,2,3,4全包), 你可以更動為 vlan304(一般網卡)
###export NCCL_SOCKET_IFNAME=vlan304 

# Singularity 容器執行上方demo.cmd
## $PWD:/DEEPSPEED 將目前目錄掛載成虛擬目錄 /DEEPSPEED
## 模型檔案位置需要掛載 /work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf

echo "
srun --mpi=pmix singularity exec --nv \
-B $PWD:/DEEPSPEED \
-B /work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf \
-B /work \
-B /work/waue0920/libraryFolder/S-waue_cuda11.8_pytorch_2.1.2_llama/local/lib:/home/waue0920/.local/lib \
-B /work/waue0920/libraryFolder/S-waue_cuda11.8_pytorch_2.1.2_llama/local/bin:/home/waue0920/.local/bin \
/home/waue0920/slurm/sif/c00cjz00_cuda11.8_pytorch_2.1.2-cuda11.8-cudnn8-devel-llama_factory.sif \
torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nnodes ${SLURM_NNODES} \
    --node_rank ${SLURM_PROCID} \
    env.py
" >> tmp.log


srun --mpi=pmix singularity exec --nv \
-B $PWD:/DEEPSPEED \
-B /work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf \
-B /work \
-B /work/waue0920/libraryFolder/S-waue_cuda11.8_pytorch_2.1.2_llama/local/lib:/home/waue0920/.local/lib \
-B /work/waue0920/libraryFolder/S-waue_cuda11.8_pytorch_2.1.2_llama/local/bin:/home/waue0920/.local/bin \
/home/waue0920/slurm/sif/c00cjz00_cuda11.8_pytorch_2.1.2-cuda11.8-cudnn8-devel-llama_factory.sif \
torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nnodes ${SLURM_NNODES} \
    --node_rank ${SLURM_PROCID} \
    env.py

