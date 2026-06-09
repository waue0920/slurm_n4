#!/bin/bash
# ── local_master_benchmark.sh ────────────────────────────────
# 手動多主機壓測 - 主節點啟動腳本 (Rank 0)

show_help() {
    echo -e "\033[1;33m[使用說明] 手動多主機壓測 - 主節點 (Master / Rank 0)\033[0m"
    echo "用法: ./local_master_benchmark.sh [DURATION] [TARGET_GB] [其他旗標]"
    echo ""
    echo "  DURATION   壓測時間(秒)  (例如: 3600 代表 1 小時)"
    echo "  TARGET_GB  GPU 額外佔用記憶體(GB) (例如: 130)"
    echo ""
    echo "常用旗標 (選填):"
    echo "  --no_cpu_stress   關閉背景 CPU 壓測"
    echo "  --cpu_cores N     自訂 CPU 壓測核心數 (預設: 90)"
    echo "  --target_ram_gb N 自訂主機 RAM 佔用大小 (預設: 64)"
    echo "  --help            顯示此說明訊息"
    echo ""
    echo "執行範例:"
    echo "  ./local_master_benchmark.sh 3600 130        # 1小時 GPU(130G)+CPU+RAM 壓測"
    echo "  ./local_master_benchmark.sh 120 0 --offline  # 2分鐘煙霧測試 (WandB離線)"
    echo "===================================================================="
}

# 防呆：如果不帶參數，或輸入了 --help，就印出使用說明並退出
if [ $# -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
    exit 0
fi

DURATION=$1
TARGET_GB=$2
shift 2 2>/dev/null

# 設定手動多節點變數
export NNODES=2
export NODE_RANK=0

# 如果使用者沒有指定 Rendezvous ID，則自動產生一個
if [ -z "$RDZV_ID" ]; then
    export RDZV_ID="bench_$(date +%s)"
fi

echo -e "\033[1;33m===================================================================="
echo -e "[手動主節點] 啟動主節點 (Rank 0 / 總共 $NNODES 節點)"
echo -e "             Rendezvous ID: $RDZV_ID"
echo -e "             請在另一台主機上，確保參數一致並啟動 local_worker_benchmark.sh"
echo -e "====================================================================\033[0m"
echo ""

# 呼叫原有主腳本，傳入 --worker 旗標以完全繞過自動 SSH 連線
./run_benchmark.sh "$DURATION" "$TARGET_GB" --worker "$@"
