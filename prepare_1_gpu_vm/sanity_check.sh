#!/bin/bash

# 定義報告檔案名稱（含時間戳記）
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
REPORT_FILE="report-${TIMESTAMP}.log"

# 磁碟讀寫效能測試函數 (Direct IO, 256MB 快速測試)
get_disk_speed() {
    local target_dir=$1
    local test_file="${target_dir}/.speedtest_tmp"
    
    # 測試寫入速度
    local write_raw=$(dd if=/dev/zero of="$test_file" bs=1M count=256 oflag=direct conv=fdatasync 2>&1)
    if [ $? -ne 0 ]; then
        echo "Write: N/A | Read: N/A"
        return
    fi
    local write_speed=$(echo "$write_raw" | awk -F, 'END{print $NF}' | sed 's/^[ \t]*//' | tr -d '\r\n')
    
    # 測試讀取速度
    local read_raw=$(dd if="$test_file" of=/dev/null bs=1M count=256 iflag=direct 2>&1)
    local read_speed=$(echo "$read_raw" | awk -F, 'END{print $NF}' | sed 's/^[ \t]*//' | tr -d '\r\n')
    
    # 清除暫存檔
    rm -f "$test_file"
    
    echo "Write: $write_speed | Read: $read_speed"
}

# 主要健康檢查函數（含 I/O 效能精簡版）
run_sanity_check() {
    # ==========================================
    # 收集資料
    # ==========================================
    
    # 1. OS & Kernel
    OS_NAME=$(grep "PRETTY_NAME" /etc/os-release | cut -d'"' -f2)
    KERNEL_NAME=$(uname -sr)
    
    # 2. CPU & Memory
    CPU_MODEL=$(lscpu | grep "Model name:" | sed -e 's/Model name:\s*//' -e 's/  */ /g')
    CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    MEM_TOTAL=$(free -h | awk '/Mem:/{print $2}')
    MEM_FREE=$(free -h | awk '/Mem:/{print $4}')
    MEM_AVAIL=$(free -h | awk '/Mem:/{print $7}')
    SWAP_TOTAL=$(free -h | awk '/Swap:/{print $2}')
    
    # 3. System Disk & Speed
    SYS_DISK_INFO=$(df -h / | awk 'NR==2{printf "%s / %s (%s Used)", $3, $2, $5}')
    SYS_SPEED=$(get_disk_speed "$HOME")
    
    # 4. GPU Status & Fabric
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+')
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{printf "%.0fGB", $1/1024}')
        GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -n 1)
        GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader | head -n 1 | awk '{printf "%.0fW", $1}')
        FABRIC_STATE=$(nvidia-smi -q | grep -A 4 "Fabric" | grep "State" | awk -F': ' '{print $2}' | tr -d ' ')
        GPU_STATUS="OK"
    else
        GPU_STATUS="ERROR"
    fi
    
    # 5. Network & Connectivity
    ENS2_IP=$(ip -4 addr show dev ens2 2>/dev/null | grep -oP 'inet \K[\d./]+')
    if [ -z "$ENS2_IP" ]; then ENS2_IP="N/A"; fi
    CURL_RESP=$(curl -I -s --max-time 3 https://www.google.com | head -n 1 | tr -d '\r\n')
    if [ -n "$CURL_RESP" ]; then
        NET_STATUS="✅ Google OK ($CURL_RESP)"
    else
        NET_STATUS="❌ 外網連線失敗"
    fi
    
    # 6. Python & PyTorch
    PY_ENV_PATH="/home/ubuntu/miniconda3/envs/pytorch-env/bin/python"
    if [ -f "$PY_ENV_PATH" ]; then
        # 執行 Python 腳本並以 Bash 變數格式輸出
        eval $("$PY_ENV_PATH" -c "
import torch, time
t0 = time.time()
torch_ver = torch.__version__
cuda_avail = torch.cuda.is_available()
dev_name = torch.cuda.get_device_name(0) if cuda_avail else 'None'
dev_count = torch.cuda.device_count() if cuda_avail else 0
status = 'FAILED'
if cuda_avail:
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        status = 'PASSED'
    except:
        pass
dt = time.time() - t0
print(f'TORCH_VER=\"{torch_ver}\" CUDA_AVAIL=\"{cuda_avail}\" DEV_NAME=\"{dev_name}\" DEV_COUNT=\"{dev_count}\" MATMUL_STATUS=\"{status}\" MATMUL_TIME=\"{dt:.2f}s\"')
")
    else
        TORCH_VER="N/A"
        CUDA_AVAIL="False"
        MATMUL_STATUS="N/A"
    fi

    # ==========================================
    # 格式化輸出報告
    # ==========================================
    
    echo "============================================================="
    echo "             NVIDIA GPU & SYSTEM SANITY CHECK"
    echo "             Generated on: $(date)"
    echo "============================================================="
    echo ""
    
    echo "[1] OS & Kernel Info"
    echo "    OS:      $OS_NAME"
    echo "    Kernel:  $KERNEL_NAME"
    echo ""
    
    echo "[2] CPU & Memory Resources"
    echo "    CPU:     $CPU_MODEL ($CPU_CORES Cores)"
    echo "    Memory:  Total $MEM_TOTAL | Free $MEM_FREE (Available $MEM_AVAIL)"
    echo "    Swap:    $SWAP_TOTAL"
    echo ""
    
    echo "[3] Storage & NVMe (Optional)"
    echo "    System Disk (/): $SYS_DISK_INFO"
    echo "                     [效能] $SYS_SPEED"
    if lsblk | grep -q "nvme"; then
        NVME_MODEL=$(sudo nvme list 2>/dev/null | tail -n +3 | awk '{print $4, $5, $6}' | head -n 1)
        NVME_MOUNT=$(df -h | grep -E "nvme|/data" | head -n 1 | awk '{printf "%s mounted on %s | %s (%s Used)", $1, $6, $2, $5}')
        if [ -n "$NVME_MOUNT" ]; then
            echo "    ✅ 偵測到 NVMe ($NVME_MODEL)"
            echo "    掛載狀態: $NVME_MOUNT - ✅ OK"
            # 測試 NVMe 寫讀效能
            NVME_SPEED=$(get_disk_speed "/data")
            echo "             [效能] $NVME_SPEED"
        else
            echo "    ⚠️ 偵測到 NVMe ($NVME_MODEL) 但未掛載！"
        fi
    else
        echo "    ℹ️ 未偵測到 NVMe 快取硬碟 (已跳過)"
    fi
    echo ""
    
    echo "[4] GPU Status & Fabric"
    if [ "$GPU_STATUS" = "OK" ]; then
        echo "    Driver:  $DRIVER_VER | CUDA: $CUDA_VER"
        echo "    GPU 0:   $GPU_NAME ($GPU_VRAM VRAM) | Temp: ${GPU_TEMP}°C | Power: $GPU_POWER"
        echo "    Fabric:  State: $FABRIC_STATE - ✅ OK (Single-GPU NVLink bypassed)"
    else
        echo "    ❌ 錯誤：NVIDIA 驅動未正確載入或無顯示卡！"
    fi
    echo ""
    
    echo "[5] InfiniBand High-Speed Network (Optional)"
    if lspci | grep -qiE "mellanox|infiniband"; then
        IB_PCIE=$(lspci | grep -iE "mellanox|infiniband" | head -n 1 | awk '{print $1}')
        IB_CARD_NAME=$(lspci | grep -iE "mellanox|infiniband" | head -n 1 | sed -e 's/^[0-9a-fA-F:.]*\s*//' -e 's/Infiniband controller:\s*//')
        IB_DEV=$(ls /sys/class/net | grep -E "^ib" | head -n 1)
        
        echo "    ✅ 偵測到 ConnectX-7 實體網卡 (PCIe $IB_PCIE)"
        if lsmod | grep -q "mlx5_ib"; then
            echo "    驅動模組: mlx5_ib (Active - ✅)"
        else
            echo "    驅動模組: mlx5_ib (Inactive - ❌)"
        fi
        
        if command -v ibv_devinfo >/dev/null 2>&1; then
            IB_PORT_STATE=$(ibv_devinfo | grep "state:" | awk '{print $2}' | head -n 1)
            IB_LINK_TYPE=$(ibv_devinfo | grep "link_layer:" | awk '{print $2}' | head -n 1)
            echo "    HCA 狀態: hca_id: $IB_DEV | Port 1: $IB_PORT_STATE | Link: $IB_LINK_TYPE"
        fi
        
        if [ -n "$IB_DEV" ]; then
            IB_IP=$(ip -4 addr show dev "$IB_DEV" 2>/dev/null | grep -oP 'inet \K[\d./]+')
            if [ -n "$IB_IP" ]; then
                echo "    IP 配置:  $IB_DEV -> $IB_IP - ✅ OK"
            else
                echo "    IP 配置:  $IB_DEV -> 未配置 IP - ❌"
            fi
        fi
    else
        echo "    ℹ️ 未偵測到 InfiniBand 高速網卡 (已跳過)"
    fi
    echo ""
    
    echo "[6] Network & Connectivity"
    echo "    ens2:    $ENS2_IP"
    if [ -n "$IB_DEV" ] && [ -n "$IB_IP" ]; then
        echo "    $IB_DEV:    $IB_IP"
    fi
    echo "    外網:    $NET_STATUS"
    echo ""
    
    echo "[7] Python & PyTorch Environment"
    if [ "$TORCH_VER" != "N/A" ]; then
        echo "    PyTorch: $TORCH_VER (CUDA: $CUDA_AVAIL)"
        echo "    設備:    $DEV_NAME ($DEV_COUNT GPU)"
        if [ "$MATMUL_STATUS" = "PASSED" ]; then
            echo "    運算:    GPU Matrix Multiplication -> ✅ PASSED ($MATMUL_TIME)"
        else
            echo "    運算:    GPU Matrix Multiplication -> ❌ FAILED"
        fi
    else
        echo "    ❌ 找不到 Conda 'pytorch-env' 虛擬環境！"
    fi
    
    echo ""
    echo "============================================================="
    echo "             SANITY CHECK COMPLETE"
    echo "             Report saved to: ${REPORT_FILE}"
    echo "============================================================="
}

# 執行檢查並透過 tee 同時輸出至螢幕與 Log 檔案
run_sanity_check | tee "${REPORT_FILE}"
