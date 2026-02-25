# N4 主機壓力測試與環境檢查專案

本專案包含兩個主要工具：用於檢查叢集環境與 GPU 配置的 `case1_envcheck`，以及執行深度效能壓測的 `case2_stresstest`。


---

## 0. 準備環境
```bash
pip install -r requirements.txt
```

## 1. 基礎環境檢查 (case1_envcheck)

在使用正式壓測前，建議先執行環境檢查，確保節點分配、GPU 資源與網路掛載正確。

### 1.1 參數調整
編輯 `case1_envcheck/check_env.sb` 修改 Slurm 資源：

```bash
#SBATCH -A GOV115010  ## !! 更換成自己的計劃代號 !! 
#SBATCH --nodes=2     ## 節點數量
#SBATCH --gres=gpu:8  ## 每個節點的 GPU 數量
```

### 1.2 送出任務
切換至 `case1_envcheck` 目錄並派送任務：
```bash
cd case1_envcheck
sbatch check_env.sb
```

### 1.3 查看輸出內容
- **紀錄檔**：預設輸出至 `case1_envcheck/check_env_sh.log`，可透過此檔案確認環境細節。
- **GPU 資源**：列出 `nvidia-smi` 偵測到的 GPU 資訊與 UUID。
- **Slurm 變數**：確認 `SLURM_JOB_GPUS` 等變數是否符合預期分配（節點數與 GPU 數）。

---

## 2. 深度壓力測試 (case2_stresstest)

用於多節點環境的效能基準試算。可同時測試 GPU 運算能力 (TFLOPS)、NCCL 網路頻寬、本地 SSD 以及 NFS 所載入的網路硬碟效能。

* 主要功能
- **GPU 運算壓測**：執行半精度矩陣乘法 (GEMM)，並記錄單張 GPU 的 TFLOPS。
- **NCCL 網通壓測**：執行 `all_reduce` 通訊，檢測節點間的網路頻寬 (GB/s)。
- **磁碟 I/O 壓測**：包含本地 SSD、Home NFS 與 Work NFS 的讀寫效能（含 Read 數據）。
- **個別 GPU 監控**：自動報告每一張卡 (Rank) 的效能，幫助快速定位故障硬體。

### 2.1 參數調整
編輯 `case2_stresstest/benchmark_v3.sb` 修改 Slurm 資源或是測試參數：

```bash
#SBATCH -A GOV115010  ## !! 更換成自己的計劃代號 !! 
#SBATCH --nodes=2     ## 節點數量
#SBATCH --gres=gpu:8  ## 每個節點的 GPU 數量

# 壓力測試超參數
DURATION=600      # 測試持續時間 (秒)
TARGET_GB=120     # 每個 GPU 填充的顯存量 (GB)
GEMM_SIZE=16384   # 矩陣乘法規模
NET_SIZE_MB=1024  # NCCL 通訊測試資料量
```

### 2.2 送出任務
```bash
sbatch case2_stresstest/benchmark_v3.sb
```

---

### 2.3 查看輸出內容

#### 紀錄檔輸出 (`<JOB_ID>_<JOB_NAME>.log`)
查看 Slurm 產生的日誌檔案（例如 `5222_h200_2n8g_v4.log`），腳本會印出 **INTERIM RESULTS** 與 **INDIVIDUAL GPU COMPUTE** 表格：
- **GPU Compute**: 全局平均 TFLOPS。
- **NFS Write/Read**: 網路硬碟吞吐量（MB/s）。
- **Individual GPU Status**: 列出每一張卡 (Rank) 的運算能力。如果某張卡數值異常（例如明顯低於同型號平均值），請檢查該節點。

```bash
=======================================================
           INTERIM RESULTS (Loop 10)
=======================================================
Metric                         | Average Value       
-------------------------------+---------------------
GPU Compute (TFLOPS)           |       660.18 TFLOPS
Network Bandwidth (GB/s)       |       189.00 GB/s
SSD Write (MB/s)               |       208.53 MB/s
SSD Read (MB/s)                |      3424.01 MB/s
Home NFS Write (MB/s)          |       340.64 MB/s
Home NFS Read (MB/s)           |      1860.95 MB/s
Work NFS Write (MB/s)          |      1304.80 MB/s
Work NFS Read (MB/s)           |      1108.06 MB/s
Avg Temp (C)                   |        51.65 C
Avg Power (W)                  |       276.12 W
Avg Util (%)                   |        99.74 %
Avg VRAM (MB)                  |    129524.95
Progress                       |  10/10 loops
=======================================================

```

#### 本地 CSV 紀錄 (`results_20260225.csv`)
包含詳細的監控數據，欄位包括：
- `timestamp`: 紀錄時間
- `hostname`: 執行節點名稱
- `GPU Compute (TFLOPS)`: 全域平均算力
- `Network Bandwidth (GB/s)`: NCCL 通訊頻寬
- `SSD Write (MB/s)`: 本地 SSD 寫入速度
- `SSD Read (MB/s)`: 本地 SSD 讀取速度
- `Home NFS Write (MB/s)`: Home 目錄寫入速度
- `Home NFS Read (MB/s)`: Home 目錄讀取速度
- `Work NFS Write (MB/s)`: Work 目錄寫入速度
- `Work NFS Read (MB/s)`: Work 目錄讀取速度
- `Avg Temp (C)`: 平均溫度
- `Avg Power (W)`: 平均功耗
- `Avg Util (%)`: 平均使用率
- `Avg VRAM (MB)`: 平均顯存佔用
- `Progress`: 測試進度
- `GPU0_TFLOPS` ~ `GPU7_TFLOPS`: 個別 GPU 的運算能力 (TFLOPS)

#### WandB 儀表板
- 監控長時間測試下的穩定度與 GPU 降頻 (Throttling) 現象。

---

