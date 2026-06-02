# N4 運算資源效能檢測工具 (N4 Benchmark Toolkit)

本專案提供一套完整的工具組，用於驗證與測試高效率運算資源（包含 NCHC N4 Slurm 叢集與 OpenStack VM 環境）的效能與穩定性。

## 專案架構

- **`slurm/`**: 專為 NCHC N4 Slurm 環境設計的檢測腳本。
  - `case1_envcheck/`: 基礎環境、GPU 分配與 Slurm 變數檢查。
  - `case2_stresstest/`: 深度壓力測試，包含 TFLOPS、NCCL 頻寬、Disk I/O 監控。
- **`openstack_vm_benchmark/`**: 針對 OpenStack 虛擬機環境的效能測試工具。
- **`setup_env.sh`**: 統一環境安裝腳本（含 Miniconda 與依賴套件）。
- **`requirements.txt`**: 專案通用的 Python 依賴套件。

---

## 1. 快速開始：環境建置 (通用)

不論是在 Slurm 還是 OpenStack VM，皆建議執行根目錄的自動化安裝腳本。此腳本會自動安裝 Miniconda (若無) 並建立名為 `n4_bench` 的獨立環境。

```bash
bash setup_env.sh
```

---

## 2. Slurm 叢集環境 (NCHC N4)

### 2.1 執行測試
1. **基礎環境檢查 (Case 1)**
   ```bash
   cd slurm/case1_envcheck
   sbatch check_env.sb
   ```

2. **深度壓力測試 (Case 2)**
   ```bash
   cd slurm/case2_stresstest
   sbatch benchmark_v3.sb
   ```

### 2.2 測試內容說明
- **GPU Compute**: 執行 GEMM 運算，計算實際 TFLOPS。
- **Network**: 使用 NCCL `all_reduce` 測試節點間頻寬 (GB/s)。
- **Storage**: 測試 Local SSD 與 NFS (Home/Work) 的讀寫效能。
- **Monitoring**: 支援 Weights & Biases (WandB) 即時監控。

---

## 3. OpenStack VM 環境

### 3.1 準備工作
1. **SSH 免密碼登入設定**
   測試前需確保多台 VM 之間可以互相免密碼登入。
   ```bash
   bash openstack_vm_benchmark/prepare_ssh.sh
   ```
2. **安裝依賴** (同第 1 節)
   在每台 VM 節點上皆執行一次：
   ```bash
   bash setup_env.sh
   ```

### 3.2 執行測試
```bash
cd openstack_vm_benchmark

# 單機測試 (Local Mode)
./run_benchmark.sh 60 80 --local

# 多機測試 (Multi-node Mode)
./run_benchmark.sh 600 80
```

---

## 4. 數據監控 (WandB)

1. **登入 WandB**
   ```bash
   conda activate n4_bench
   wandb login
   ```
2. **查看報告**: 測試執行中會輸出 WandB 連結，可即時查看 GPU 功耗、溫度、算力等趨勢圖。

---

## 注意事項
- 執行 Slurm 腳本前，請務必修改 `.sb` 檔案中的 `#SBATCH -A <PLAN_ID>`。
- 專案依賴已整合於 `n4_bench` 環境中，手動執行 Python 腳本前請先 `conda activate n4_bench`。
