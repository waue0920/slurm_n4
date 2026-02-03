#!/bin/bash

# 切換到工作目錄
WORKDIR=/home/waue0920/yolov9
BATCH_NUM=64
cd $WORKDIR

# 設置必要的環境變數
export NCCL_DEBUG=INFO

# 從 SLURM 環境變數抓取參數
GPUS_NUM=$(nvidia-smi -L | wc -l)  # 等於 $SLURM_GPUS_ON_NODE
DEVICE_LIST=$(seq -s, 0 $((GPUS_NUM-1)) | paste -sd, -) # 0,1,...n-1
NNODES=${SLURM_NNODES:-1}               # 節點總數，默認為 1
NODE_RANK=${SLURM_NODEID}            # 當前節點的 rank，默認為 0

# 確定 MASTER_ADDR 無法slurm參數來，需要透過 sbatch.sb 去 export 參數過來
MASTER_PORT=9527  # 固定的通信端口

# 列印參數
echo "Running on node $(hostname)"
echo "SLURM_NODEID=$NODE_RANK"
echo "SLURM_NNODES=$NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# 訓練命令 (動態設置 nproc_per_node 和 nnodes)
TRAIN_CMD="torchrun --nproc_per_node=$GPUS_NUM --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         segment/train_dual.py --workers 32 --device $DEVICE_LIST --batch $BATCH_NUM \
         --data coco.yaml --img 640 --cfg models/segment/yolov9-c-dseg.yaml \
         --weights '' --name gelan-c-seg --hyp hyp.scratch-high.yaml --no-overlap \
         --epochs 200 --close-mosaic 10"


# 印即將執行的命令行
echo "Executing command: $TRAIN_CMD"

# 執行訓練命令，並檢查是否成功
$TRAIN_CMD
if [ $? -ne 0 ]; then
	echo "Error: TRAIN_CMD execution failed on node $(hostname)" >&2
	exit 1
fi
