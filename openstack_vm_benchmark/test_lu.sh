#!/bin/bash
# run_benchmark.sh  –  Multi-node GPU benchmark launcher
# Usage: ./run_benchmark.sh [DURATION] [TARGET_GB] [FLAGS]

# ── Configuration ────────────────────────────────────────────
MASTER_ADDR="192.168.230.136"   # vm201
REMOTE_NODES=("vm202")
MASTER_PORT="29500"
NPROC_PER_NODE=8
USER="ubuntu"
WORKSPACE="/home/ubuntu/workspace/slurm_n4/openstack_vm_benchmark"
SCRIPT_PATH="$WORKSPACE/benchmark_lib_lu.py"

# ── Help ─────────────────────────────────────────────────────
show_help() {
    echo "Usage: ./run_benchmark.sh [DURATION] [TARGET_GB] [FLAGS]"
    echo ""
    echo "Positional Arguments:"
    echo "  DURATION    Test duration in seconds (default: 86400 = 24 h)"
    echo "  TARGET_GB   VRAM to fill per GPU in GB (default: 80)"
    echo ""
    echo "Flags:"
    echo "  --local     Run only on the current node (single-node test)"
    echo "  --offline   Run WandB in offline mode"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_benchmark.sh 86400 80            # 24-hour multi-node"
    echo "  ./run_benchmark.sh 3600 80             # 1-hour multi-node"
    echo "  ./run_benchmark.sh 60 0 --local        # quick single-node smoke test"
}

# ── Defaults ─────────────────────────────────────────────────
DURATION=86400    # 24 hours
TARGET_GB=80
MODE="MULTI"
IS_WORKER=false
OFFLINE_FLAG=""

# ── Arg parsing ──────────────────────────────────────────────
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)    show_help; exit 0 ;;
        --local)   MODE="LOCAL"; shift ;;
        --offline) export WANDB_MODE=offline; OFFLINE_FLAG="--offline"; shift ;;
        --worker)  IS_WORKER=true; shift ;;
        *)         ARGS+=("$1"); shift ;;
    esac
done

[ ${#ARGS[@]} -ge 1 ] && DURATION=${ARGS[0]}
[ ${#ARGS[@]} -ge 2 ] && TARGET_GB=${ARGS[1]}

# ── Conda environment ─────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate n4_bench
fi

# ── NCCL / IB environment ────────────────────────────────────
export WANDB_MODE=${WANDB_MODE:-online}

# InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET="IB"
export NCCL_NET_GDR_LEVEL=0           # disable GPUDirect RDMA if unstable; set to 2 to enable
export NCCL_SOCKET_IFNAME=ens2
export NCCL_DEBUG=WARN                # use INFO for deep debugging, WARN for 24h runs

# IB timeouts / retries  (prevents spurious failures over long runs)
export NCCL_IB_TIMEOUT=22            # 2^22 * 4.096 µs ≈ 17 s per retry
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4  # more QPs = more bandwidth utilisation

# NVSwitch / NVLink
export NCCL_P2P_LEVEL=NVL            # maximise H200 NVSwitch bandwidth

# Resiliency
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF   # set to DETAIL only when debugging

# Detect IB HCAs
IB_DEVICES=$(ibv_devinfo 2>/dev/null | grep hca_id | awk '{print $2}' | tr '\n' ',' | sed 's/,$//')
if [ -n "$IB_DEVICES" ]; then
    export NCCL_IB_HCA=$IB_DEVICES
    echo "[ENV] IB HCAs: $NCCL_IB_HCA"
else
    echo "[WARN] No IB devices found – NCCL will fall back to TCP"
fi

# ── Identify node rank ───────────────────────────────────────
# Try ens2 first, fall back to hostname -I
MY_IP=$(ip -4 addr show ens2 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
[ -z "$MY_IP" ] && MY_IP=$(hostname -I | awk '{print $1}')

if [ "$MY_IP" == "$MASTER_ADDR" ]; then
    NODE_RANK=0
    NNODES=$((1 + ${#REMOTE_NODES[@]}))
else
    NODE_RANK=1
    NNODES=2
fi

# ── Local-mode overrides ─────────────────────────────────────
if [ "$MODE" == "LOCAL" ]; then
    NNODES=1
    NODE_RANK=0
    # Disable IB for single-node to avoid loopback crashes
    unset NCCL_NET
    export NCCL_IB_DISABLE=1
fi

echo "[LAUNCH] Mode=$MODE | Rank=$NODE_RANK/$NNODES | Master=$MASTER_ADDR | IB=${NCCL_IB_HCA:-none} | WandB=$WANDB_MODE | Duration=${DURATION}s | TargetGB=${TARGET_GB}"

# ── Spawn workers via SSH (master only, not a --worker call) ─
if [ "$MODE" == "MULTI" ] && [ "$NODE_RANK" == "0" ] && [ "$IS_WORKER" == "false" ]; then
    for node in "${REMOTE_NODES[@]}"; do
        echo "[LAUNCH] Starting worker on $node …"
        ssh -o BatchMode=yes \
            -o ConnectTimeout=10 \
            $USER@$node \
            "cd $WORKSPACE && ./test_lu.sh $DURATION $TARGET_GB --worker $OFFLINE_FLAG" &
    done
    # Wait a bit longer than default to let workers initialise NCCL
    sleep 5
fi

# ── torchrun ─────────────────────────────────────────────────
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    "$SCRIPT_PATH" \
        --duration $DURATION \
        --target_gb $TARGET_GB \
        $OFFLINE_FLAG

EXIT_CODE=$?
wait

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] torchrun exited with code $EXIT_CODE"
else
    echo "[DONE] Benchmark finished cleanly."
fi

exit $EXIT_CODE
