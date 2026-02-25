# N4 主機壓力測試專案

這是一個用於多節點、多 GPU 環境的壓力測試與效能基準試算腳本。它可以同時測試 GPU 運算能力 (TFLOPS)、NCCL 網路頻寬、本地 SSD 以及 NFS 所載入的網路硬碟效能。

## 主要功能
- **GPU 運算壓測**：執行半精度矩陣乘法 (GEMM)，並記錄單張 GPU 的 TFLOPS。
- **NCCL 網通壓測**：執行 `all_reduce` 通訊，檢測節點間的網路頻寬 (GB/s)。
- **磁碟 I/O 壓測**：
  - **本地 SSD**: `/tmp` 路徑讀寫。
  - **Network NFS**: Home 與 Work 路徑的讀寫效能 (MB/s)。
- **個別 GPU 監控**：自動識別並報告每一張 GPU 的效能，幫助快速定位「掉隊」的節點或損壞的顯卡。
- **即時回傳**：支援 WandB 線上圖表監控與 CSV 本地記錄。

---

## 如何使用

### 1. 準備環境
建議使用 Conda 或 Virtualenv 環境，並安裝必要套件：
```bash
pip install -r requirements.txt
```

### 2. 修改排程腳本
編輯 `benchmark_v3.sb` (或現有的 `.sb` 檔案)，確保 `torchrun` 指向的是 `stress_test_v4.py`：

```bash
# benchmark_v3.sb 範例片段
srun torchrun \
    ... \
    ./stress_test_v4.py \
    --project $PROJECT \
    --duration $DURATION \
    ...
```

### 3. 使用 Slurm 派送任務
使用 `sbatch` 指令將任務派送到排程器：

```bash
sbatch benchmark_v3.sb
```

---

## 如何閱讀結果

### 1. 終端機輸出 (Output Log)
腳本每隔一段時間會印出 **INTERIM RESULTS** 表格：
- **GPU Compute**: 全局平均 TFLOPS。
- **NFS Write/Read**: 網路硬碟吞吐量（包含 Read 數據）。
- **INDIVIDUAL GPU COMPUTE**: 列出每一張 Rank 的運算能力。如果某張 Rank 的數值顯著低於其他卡，代表該節點可能有硬體問題。

### 2. 本地 CSV 紀錄 (`results_20250225.csv`)
每一行代表一次報告週期的快照，欄位包含：
- `timestamp`: 時間戳記。
- `GPU Compute (TFLOPS)`: 整體平均值。
- `Rank_N_TFLOPS`: **第 N 張卡的獨立效能**。
- `Home/Work NFS Read/Write`: 硬碟讀寫詳細數據。

### 3. WandB 儀表板
- **Time Series**: 觀察測試過程中效能是否隨溫度升高而有降頻 (Throttling) 現象。
- **Summary**: 測試結束後的最終平均效能統計。

---

## 注意事項
- **清除檔案**：腳本執行的過程中會產生臨時檔 (.tmp)，正常結束時會自動刪除；若因崩潰中止，請手動清理各路徑下的 `stress_*.tmp`。
- **權限**：請確保 `--home_dir` 與 `--work_dir` 是你有權限讀寫的路徑。
