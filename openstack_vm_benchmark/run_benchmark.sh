#!/bin/bash
# run_benchmark.sh  –  Multi-node GPU stress benchmark launcher

# ── Configuration ────────────────────────────────────────────
MASTER_ADDR="192.168.230.136"
REMOTE_NODES=("vm202")
MASTER_PORT="29500"
NPROC_PER_NODE=8
USER="ubuntu"
WORKSPACE="/home/ubuntu/workspace/slurm_n4/openstack_vm_benchmark"
SCRIPT_PATH="$WORKSPACE/benchmark_lib.py"

# ── Help ─────────────────────────────────────────────────────
show_help() {
    echo "Usage: ./run_benchmark.sh [DURATION] [TARGET_GB] [FLAGS]"
    echo ""
    echo "  DURATION   seconds  (default: 86400 = 24 h)"
    echo "  TARGET_GB  extra VRAM filler per GPU in GB (default: 0)"
    echo ""
    echo "  --local    single-node only"
    echo "  --offline  WandB offline mode"
    echo "  --help     this message"
    echo ""
    echo "Examples:"
    echo "  ./run_benchmark.sh 86400 0          # 24h multi-node, no filler"
    echo "  ./run_benchmark.sh 3600  0          # 1h  multi-node"
    echo "  ./run_benchmark.sh 120   0 --local  # 2min smoke test"
}

# ── Defaults ─────────────────────────────────────────────────
DURATION=86400
TARGET_GB=0
MODE="MULTI"
IS_WORKER=false
OFFLINE_FLAG=""

# ── Arg parsing ──────────────────────────────────────────────
if [ $# -eq 0 ]; then show_help; exit 0; fi

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

# ── Conda ────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && {
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate n4_bench
}

# ── NCCL / IB environment ────────────────────────────────────
export WANDB_MODE=${WANDB_MODE:-online}

# InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET="IB"
#export NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_IFNAME=ens2
export NCCL_DEBUG=WARN

# Detect IB HCAs
IB_DEVICES=$(ibv_devinfo 2>/dev/null | grep hca_id | awk '{print $2}' \
             | tr '\n' ',' | sed 's/,$//')
if [ -n "$IB_DEVICES" ]; then
    export NCCL_IB_HCA=$IB_DEVICES
    echo "[ENV] IB HCAs: $NCCL_IB_HCA"
else
    echo "[WARN] No IB devices detected – NCCL will fall back to TCP"
fi

# ── Node rank detection ───────────────────────────────────────
MY_IP=$(ip -4 addr show ens2 2>/dev/null \
        | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
[ -z "$MY_IP" ] && MY_IP=$(hostname -I | awk '{print $1}')

if [ "$MY_IP" == "$MASTER_ADDR" ]; then
    NODE_RANK=0
    NNODES=$((1 + ${#REMOTE_NODES[@]}))
else
    NODE_RANK=1
    NNODES=2
fi

# ── Local-mode overrides ──────────────────────────────────────
if [ "$MODE" == "LOCAL" ]; then
    NNODES=1
    NODE_RANK=0
    unset NCCL_NET
    export NCCL_IB_DISABLE=1
fi

# ── Unique rendezvous ID (prevents stale rdzv state) ─────────
# If RDZV_ID is not already set (e.g., by the Master's SSH command), generate it
if [ -z "$RDZV_ID" ]; then
    RDZV_ID="bench_$(date +%s)"
fi

echo "[LAUNCH] Mode=$MODE | Rank=$NODE_RANK/$NNODES | Master=$MASTER_ADDR"
echo "         IB=${NCCL_IB_HCA:-none} | rdzv_id=$RDZV_ID"
echo "         Duration=${DURATION}s | TargetGB=${TARGET_GB}"

# ── SSH worker spawn (master only) ───────────────────────────
if [ "$MODE" == "MULTI" ] && [ "$NODE_RANK" == "0" ] && [ "$IS_WORKER" == "false" ]; then
    WORKER_RANK=1
    for node in "${REMOTE_NODES[@]}"; do
        echo "[LAUNCH] Spawning worker on $node (Rank $WORKER_RANK/$NNODES) …"
        # Pass RDZV_ID and node topology info directly into the remote execution environment
        ssh -o BatchMode=yes -o ConnectTimeout=10 \
            $USER@$node \
            "cd $WORKSPACE && RDZV_ID=$RDZV_ID NODE_RANK=$WORKER_RANK NNODES=$NNODES ./run_benchmark.sh $DURATION $TARGET_GB --worker $OFFLINE_FLAG" &
        WORKER_RANK=$((WORKER_RANK + 1))
    done
    sleep 5   # give workers time to reach torchrun before master connects
fi

# ── torchrun ──────────────────────────────────────────────────
# --rdzv_backend=c10d  →  all ranks must register before ANY rank starts work.
# This eliminates the "rank 0 runs alone" bug seen in the original script.
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$RDZV_ID \
    "$SCRIPT_PATH" \
        --duration   $DURATION \
        --target_gb  $TARGET_GB \
        $OFFLINE_FLAG

EXIT=$?
wait

[ $EXIT -eq 0 ] && echo "[DONE] Benchmark finished cleanly." \
               || echo "[ERROR] torchrun exited $EXIT"
exit $EXIT
