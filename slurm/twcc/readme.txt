#!/bin/bash
# == practice == # 

### 1
srun -N 1 -n 1 --gpus-per-node=1 --account=GOV109016 --mpi=pmix singularity run --nv /home/waue0920/slurm/pytorch_22.12-py3.sif torchrun ./env.py

### 2 
squeue -u $USER 
scancel <id>

### 3 
sbatch slurm3.sb

### 4 交互 還沒成功
srun -N 1 -n 1 --gpus-per-node=2 --account=GOV109016
singularity run --nv /home/waue0920/slurm/pytorch_22.12-py3.sif
torchrun --master_addr gn0809 --master_port 11234  env.py

# == detail == #
## squeue ：查看 partition 狀態
squeue -u $USER

## sacct : 查看自己的job
sacct 

## scontrol  : 可以查別人的job，也可以看node上的狀態
scontrol show job <id>
scontrol show node gn0101.twcc.ai

## sinfo ： 查看 partition 與node的狀態
sinfo

## sbatch 
sbatch slurm1.sb

## srun
srun --gpus-per-node=1 date

## salloc 要一塊資源
salloc -N 1 -n 1 --gpus-per-node=1 --account=GOV109016

alloc -N 1 -n 1 --gpus-per-node=1 --account=GOV109016 -w gn1025
ssh gn1025

## scancel 取消資源
scancel <id>


## module load 

ml singularity
ml purge
ml av
ml 
