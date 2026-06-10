#!/bin/bash
# ── local_worker_benchmark.sh ────────────────────────────────
# 手動多主機壓測 - 工作節點啟動腳本 (Rank 1)

# ── 使用者自訂環境變數 ────────────────────────────────────────────
MASTER_ADDR="vm201"              # 主節點 IP / Hostname (必須與 Master 相同)
MASTER_PORT="29500"              # PyTorch 通訊埠
NPROC_PER_NODE=8                 # 每個節點的 GPU 數量
NCCL_SOCKET_IFNAME="ens2"        # 網路介面名稱 (例如 ens2, ib0, ens2f0np0 等)
WORKSPACE="/home/ubuntu/workspace/slurm_n4/openstack_vm_benchmark"
# ─────────────────────────────────────────────────────────────

show_help() {
    echo -e "\033[1;36m[使用說明] 手動多主機壓測 - 工作節點 (Worker / Rank 1)\033[0m"
    echo "用法: ./local_worker_benchmark.sh [DURATION] [TARGET_GB] [其他旗標]"
    echo ""
    echo "  DURATION   壓測時間(秒)  (請與 Master 保持完全一致！預設: 86400)"
    echo "  TARGET_GB  GPU 額外佔用記憶體(GB) (請與 Master 保持完全一致！預設: 0)"
    echo ""
    echo "常用旗標 (選填):"
    echo "  --offline         WandB 離線模式"
    echo "  --no_cpu_stress   關閉背景 CPU 壓測 (本版本預設已關閉)"
    echo "  --cpu_cores N     自訂 CPU 壓測核心數"
    echo "  --target_ram_gb N 自訂主機 RAM 佔用大小"
    echo "  --help            顯示此說明訊息"
}

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
    exit 0
fi

DURATION=${1:-86400}
TARGET_GB=${2:-0}
shift 2 2>/dev/null

# 設定手動多節點變數
export NNODES=2
export NODE_RANK=1
export RDZV_ID="bench_${MASTER_ADDR}"

# ── Conda ────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && {
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate n4_bench
}

# ── NCCL / IB environment ────────────────────────────────────
export WANDB_MODE=${WANDB_MODE:-online}
export NCCL_IB_DISABLE=0
export NCCL_NET="IB"
export NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
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

echo -e "\033[1;36m===================================================================="
echo -e "[手動工作節點] 啟動工作節點 (Rank 1 / 總共 $NNODES 節點)"
echo -e "               Master Address: $MASTER_ADDR"
echo -e "               Rendezvous ID:  $RDZV_ID"
echo -e "               Interface:      $NCCL_SOCKET_IFNAME"
echo -e "====================================================================\033[0m"
echo ""

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$RDZV_ID \
    "$WORKSPACE/benchmark_lib.py" \
        --duration   $DURATION \
        --target_gb  $TARGET_GB \
        "$@"

EXIT=$?
exit $EXIT
