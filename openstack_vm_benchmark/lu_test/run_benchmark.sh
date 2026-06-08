#!/bin/bash

# --- Configuration ---
MASTER_ADDR="192.168.230.136" # vm201
REMOTE_NODES=("vm202")        # List of worker nodes
MASTER_PORT="29500"
NPROC_PER_NODE=8
USER="ubuntu"
WORKSPACE="/home/ubuntu/workspace/slurm_n4/openstack_vm_benchmark"
SCRIPT_PATH="$WORKSPACE/benchmark_lib.py"

# --- Help Message ---
show_help() {
    echo "Usage: ./run_benchmark.sh [DURATION] [TARGET_GB] [FLAGS]"
    echo ""
    echo "Positional Arguments:"
    echo "  DURATION    Test duration in seconds (default: 60)"
    echo "  TARGET_GB   VRAM to fill per GPU in GB (default: 40)"
    echo ""
    echo "Flags:"
    echo "  --local     Run only on the current node (Single-node test)"
    echo "  --offline   Run WandB in offline mode"
    echo "  --help      Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run_benchmark.sh 60 40            # Multi-node (vm201 + vm202)"
    echo "  ./run_benchmark.sh 60 40 --local    # Single-node quick test"
}

# --- Default Values ---
DURATION=60
TARGET_GB=40
MODE="MULTI"
IS_WORKER=false
OFFLINE_FLAG=""

# --- Simple Arg Parsing ---
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --local)
            MODE="LOCAL"
            shift
            ;;
        --offline)
            export WANDB_MODE=offline
            OFFLINE_FLAG="--offline"
            shift
            ;;
        --worker)
            # Internal flag used by master to trigger workers
            IS_WORKER=true
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Assign positional args if provided
if [ ${#ARGS[@]} -ge 1 ]; then DURATION=${ARGS[0]}; fi
if [ ${#ARGS[@]} -ge 2 ]; then TARGET_GB=${ARGS[1]}; fi

# --- Environment ---
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate n4_bench
fi

export WANDB_MODE=${WANDB_MODE:-online}
export NCCL_SOCKET_IFNAME=ens2
export NCCL_NET="IB"
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --- Resiliency and NVSwitch Safeguards ---
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22  # Increases the InfiniBand hardware retry timeout
export NCCL_IB_RETRY_CNT=7
export NCCL_P2P_LEVEL=NVL  # Maximizes H200 NVSwitch performance

# Detect IB devices
IB_DEVICES=$(ibv_devinfo | grep hca_id | awk '{print $2}' | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA=$IB_DEVICES

# --- Identify Node Rank ---
MY_IP=$(ip -4 addr show ens2 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
if [ "$MY_IP" == "$MASTER_ADDR" ]; then
    NODE_RANK=0
    NNODES=$((1 + ${#REMOTE_NODES[@]}))
else
    # In this simple 2-node setup, any non-master is rank 1
    NODE_RANK=1
    NNODES=2
fi

# --- Local Fallback Logic ---
if [ "$MODE" == "LOCAL" ]; then
    NNODES=1
    NODE_RANK=0
    # Prevent InfiniBand loopback crashes when testing on a single node
    unset NCCL_NET
fi

echo "[LAUNCH] Mode: $MODE | Rank: $NODE_RANK/$NNODES | Master: $MASTER_ADDR | IB: $NCCL_IB_HCA | WandB: $WANDB_MODE"

# --- Remote Launch Logic ---
if [ "$MODE" == "MULTI" ] && [ "$NODE_RANK" == "0" ] && [ "$IS_WORKER" == "false" ]; then
    for node in "${REMOTE_NODES[@]}"; do
        echo "[LAUNCH] Starting worker on $node..."
        ssh -o BatchMode=yes $USER@$node "cd $WORKSPACE && ./run_benchmark.sh $DURATION $TARGET_GB --worker $OFFLINE_FLAG" &
    done
    sleep 2
fi

# --- Execution ---
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    $SCRIPT_PATH --duration $DURATION --target_gb $TARGET_GB $OFFLINE_FLAG

wait
