#!/bin/bash
#SBATCH --job-name=Hello_twcc    ## job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waue0920@gmail.com
#SBATCH --nodes=2                ## 索取 2 節點
#SBATCH --ntasks-per-node=2      ## 每個節點運行 2 srun tasks
#SBATCH --cpus-per-task=2        ## 每個 srun task 索取 2 CPUs
#SBATCH --gres=gpu:2             ## 每個節點索取 2 GPUs
#SBATCH --account="GOV109016"  ## iService_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gtest        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)

module purge

SIF=/home/waue0920/slurm/pytorch_22.01-py3.sif
SINGULARITY="singularity run --nv $SIF"

# pytorch horovod benchmark script from
# wget https://raw.githubusercontent.com/horovod/horovod/v0.20.3/examples/pytorch/pytorch_synthetic_benchmark.py
PTH_RUN="python train.py --batch-size 256 "

# enable NCCL log
export NCCL_DEBUG=INFO

srun --mpi=pmix $SINGULARITY $PTH_RUN
