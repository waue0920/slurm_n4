#!/bin/bash
#SBATCH --job-name=4node_twcc    ## job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waue0920@gmail.com
#SBATCH --nodes=2             ## 索取 2 節點
#SBATCH --ntasks-per-node=2      ## 每個節點運行 2 srun tasks
#SBATCH --cpus-per-task=2        ## 每個 srun task 索取 2 CPUs
#SBATCH --gres=gpu:2             ## 每個節點索取 2 GPUs
#SBATCH --account="GOV109016"  ## iService_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gtest        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)


#srun hostname
MASTER=`/bin/hostname -s`
srun --mpi=pmix /bin/hostname -s > hosts-tmp
sort hosts-tmp > hosts-list
sleep 5
HOST=($(cat hosts-list))
echo "Mmaster is $MASTER"
echo "Multi nodes are ${HOST[*]}"

if [ $MASTER == ${HOST[0]} ]; then
    srun --mpi=pmix $1 ${HOST[*]}
fi

