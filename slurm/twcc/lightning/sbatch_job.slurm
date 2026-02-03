#!/bin/bash
#SBATCH --job-name=twcc    ## job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waue0920@gmail.com
#SBATCH --nodes=1                ## 索取  節點
#SBATCH --ntasks-per-node=1      ## 每個節點運行  srun tasks
#SBATCH --cpus-per-task=2        ## 每個 srun task 索取  CPUs
#SBATCH --gpus-per-node=2        ## 每個節點索取  GPUs  =~ --gres=gpu:2
#SBATCH --account="GOV109016"  ## iService_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)

#module load singularity

SIF=/work/waue0920/open_access/ngc2405-nemo-20240829.sif
SINGULARITY="singularity exec --nv $SIF"
PTH_RUN="python test_1node_2gpu.py"

export NCCL_DEBUG=INFO

srun --mpi=pmix $SINGULARITY $PTH_RUN
