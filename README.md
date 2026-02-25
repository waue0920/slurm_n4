# N4 主機壓力測試與環境檢查專案

本專案包含兩個主要工具：用於檢查叢集環境與 GPU 配置的 `case1_envcheck`，以及執行深度效能壓測的 `case2_stresstest`。

---

## 1. 基礎環境檢查 (case1_envcheck)

在使用正式壓測前，建議先執行環境檢查，確保節點分配、GPU 資源與網路掛載正確。

### 如何使用
切換至 `case1_envcheck` 目錄並派送任務：
```bash
cd case1_envcheck
sbatch check_env.sb
```

### 檢查內容
- **GPU 資源**：列出 `nvidia-smi` 偵測到的 GPU 資訊與 UUID。
- **Slurm 變數**：確認 `SLURM_JOB_GPUS` 等變數是否符合預期分配（節點數與 GPU 數）。

---

## 2. 深度壓力測試 (case2_stresstest)

用於多節點環境的效能基準試算。可同時測試 GPU 運算能力 (TFLOPS)、NCCL 網路頻寬、本地 SSD 以及 NFS 所載入的網路硬碟效能。

### 主要功能
- **GPU 運算壓測**：執行半精度矩陣乘法 (GEMM)，並記錄單張 GPU 的 TFLOPS。
- **NCCL 網通壓測**：執行 `all_reduce` 通訊，檢測節點間的網路頻寬 (GB/s)。
- **磁碟 I/O 壓測**：包含本地 SSD、Home NFS 與 Work NFS 的讀寫效能（含 Read 數據）。
- **個別 GPU 監控**：自動報告每一張卡 (Rank) 的效能，幫助快速定位故障硬體。

### 如何使用

#### 1. 準備環境
```bash
pip install -r requirements.txt
```

#### 2. 修改排程腳本
編輯 `case2_stresstest/benchmark_v3.sb` (或 `benchmark_v2.sb`)，確保 `torchrun` 指向的是 `stress_test_v4.py`：

```bash
# case2_stresstest/benchmark_v3.sb 範例
srun torchrun \
    ... \
    ./stress_test_v4.py \
    --project $PROJECT \
    --duration $DURATION \
    --target_gb 120
```

#### 3. 派送任務
```bash
sbatch case2_stresstest/benchmark_v3.sb
```

---

## 📊 如何閱讀結果

### 1. 終端機輸出 (Output Log)
腳本會印出 **INTERIM RESULTS** 與 **INDIVIDUAL GPU COMPUTE** 表格：
- **GPU Compute**: 全局平均 TFLOPS。
- **NFS Write/Read**: 網路硬碟吞吐量（MB/s）。
- **Individual GPU Status**: 列出每一張卡 (Rank) 的運算能力。如果某張卡數值異常（例如明顯低於同型號平均值），請檢查該節點。

### 2. 本地 CSV 紀錄 (`case2_stresstest/results_20250225.csv`)
包含所有詳細數據，包含 `timestamp`、整體平均值以及 `Rank_N_TFLOPS`（各別 GPU 效能）。

### 3. WandB 儀表板
- 監控長時間測試下的穩定度與 GPU 降頻 (Throttling) 現象。

---

## 注意事項
- **清除檔案**：腳本會產生臨時檔 (.tmp)，正常結束時會自動刪除；若因崩潰中止，請手動清理各路徑下的 `stress_*.tmp`。
- **路徑權限**：請確保 `--home_dir` 與 `--work_dir` 是你有權限讀寫的路徑。
