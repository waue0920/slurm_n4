#!/bin/bash

# 預設值
DEFAULT_NODES=1
DEFAULT_GPUS=8
DEFAULT_CPUS=16
ACCOUNT="GOV113038"
PARTITION="gtest"

# 幫助訊息
function usage() {
    echo "用法: $0 -n <節點數量> -g <GPU數量> -c <CPU數量>"
    echo "  -n  節點數量 (預設: $DEFAULT_NODES)"
    echo "  -g  GPU數量 (預設: $DEFAULT_GPUS)"
    echo "  -c  每任務 CPU 數量 (預設: $DEFAULT_CPUS)"
    exit 1
}

# 參數解析
while getopts "n:g:c:" opt; do
    case $opt in
        n) NODES=$OPTARG ;;
        g) GPUS=$OPTARG ;;
        c) CPUS=$OPTARG ;;
        *) usage ;;
    esac
done

# 如果參數未設置，使用預設值
NODES=${NODES:-$DEFAULT_NODES}
GPUS=${GPUS:-$DEFAULT_GPUS}
CPUS=${CPUS:-$DEFAULT_CPUS}

# 檢查是否有必要參數
if [[ -z $NODES || -z $GPUS || -z $CPUS ]]; then
    usage
fi

# 執行 salloc
salloc --nodes=$NODES --gres=gpu:$GPUS --cpus-per-task=$CPUS -A $ACCOUNT -p $PARTITION

