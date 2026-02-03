#!/bin/bash
#SBATCH -J BLOOM                   # Job name
#SBATCH -o slurm-%j.out        # Name of stdout output file (%j expands to jobId)
#SBATCH --account="GOV109016"        # iService Project id
#SBATCH --nodes=2                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of MPI process per node
#SBATCH --cpus-per-task=2          # Number of CPUs per node
#SBATCH --gres=gpu:1               # Number of GPUs per node
#SBATCH --partition=gtest          # gtest,gp1d, gp2d, gp4d

#srun hostname

module purge
module load singularity

srun --mpi=pmix /bin/hostname -s > hosts-tmp  #回傳master node
sort hosts-tmp > hosts-list
sleep 5
HOSTLIST=($(cat hosts-list))
echo "Mmaster is ${HOSTLIST[0]}" >> ./tmp-msg
echo "Multi nodes are ${HOSTLIST[*]}"  >> ./tmp-msg

SIF=/work/TWCC_cntr/pytorch_22.01-py3.sif
SINGULARITY="singularity run --nv $SIF"

MASTER_ADDR=${HOSTLIST[0]}
MASTER_PORT=6777
N_GPUS=2
NNODES=4

export LAUNCHER="torchrun --nproc_per_node $N_GPUS --nnodes $NNODES \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT "
export CMD="train.py"


echo "srun $SINGULARITY $LAUNCHER $CMD" >>./tmp-msg
srun $SINGULARITY $LAUNCHER $CMD
