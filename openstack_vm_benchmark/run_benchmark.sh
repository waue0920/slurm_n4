#!/bin/bash
# ── run_benchmark.sh ──────────────────────────────────────────
# 自動多主機壓測啟動協調器 (VM 環境一鍵啟動)

# ── 使用者自訂環境變數 ────────────────────────────────────────────
MASTER_ADDR="vm205"              # 主節點 IP / Hostname
REMOTE_NODES=("vm205-1" "vm205-2" "vm205-3" "vm205-4" "vm205-5" "vm205-6" "vm205-7")           # 遠端 Worker 節點清單
USER="ubuntu"                    # SSH 使用者名稱
WORKSPACE="/home/ubuntu/workspace/slurm_n4/openstack_vm_benchmark"
# ─────────────────────────────────────────────────────────────

show_help() {
    echo "Usage: ./run_benchmark.sh [DURATION] [TARGET_GB] [FLAGS]"
    echo ""
    echo "  DURATION   seconds  (default: 86400 = 24 h)"
    echo "  TARGET_GB  extra VRAM filler per GPU in GB (default: 0)"
    echo ""
    echo "Examples:"
    echo "  ./run_benchmark.sh 86400 130        # GPU/CPU/RAM full multi-node test"
    echo "  ./run_benchmark.sh 120   0          # 2min smoke test"
}

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
    exit 0
fi

DURATION=${1:-86400}
TARGET_GB=${2:-0}
shift 2 2>/dev/null

# ── 計算節點總數 ──────────────────────────────────────────────
TOTAL_NODES=$((1 + ${#REMOTE_NODES[@]}))

# ── 檢查節點數是否為雙數 ──────────────────────────────────────
if [ $((TOTAL_NODES % 2)) -ne 0 ]; then
    echo -e "\033[1;31m[ERROR] 執行節點總數 ($TOTAL_NODES) 非雙數！P2P 測試需要雙數節點避免死鎖。\033[0m"
    echo "        請調整 REMOTE_NODES 確保（Master + Worker）總數為雙數。"
    exit 1
fi

echo "[ORCHESTRATOR] Starting benchmark on Master and Worker nodes..."
echo "               Master Node: $MASTER_ADDR"
echo "               Worker Nodes: ${REMOTE_NODES[*]}"
echo "               Total Nodes:  ${TOTAL_NODES}"
echo "               Duration:     ${DURATION}s"
echo "               Target GB:    ${TARGET_GB}"

# ── 透過 SSH 啟動遠端 Worker ─────────────────────────────────
rank=1
for node in "${REMOTE_NODES[@]}"; do
    echo "[ORCHESTRATOR] Spawning local_worker_benchmark.sh on $node (Rank $rank) ..."
    ssh -o BatchMode=yes -o ConnectTimeout=10 \
        $USER@$node \
        "cd $WORKSPACE && NODE_RANK=$rank NNODES=$TOTAL_NODES ./local_worker_benchmark.sh $DURATION $TARGET_GB $@" &
    rank=$((rank + 1))
done

# 給 Worker 啟動並等待 Torchrun 初始化連接埠的時間
sleep 5

# ── 啟動本地 Master ──────────────────────────────────────────
echo "[ORCHESTRATOR] Running local_master_benchmark.sh locally..."
NNODES=$TOTAL_NODES ./local_master_benchmark.sh $DURATION $TARGET_GB "$@"

EXIT=$?
wait

[ $EXIT -eq 0 ] && echo "[DONE] Orchestrated benchmark finished cleanly." \
               || echo "[ERROR] Master exited with code $EXIT"
exit $EXIT
