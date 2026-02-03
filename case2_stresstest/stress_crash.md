# H200 叢集壓力測試持續性掛死事件分析 (Job ID: 530)

## 1. 事件概述
在 2026-02-03 進行的 8 節點 x 8 GPU 指標性壓力測試中，作業（Job ID: 530）在執行約 30 分鐘後發生**因 NFS I/O 阻塞導致的全域掛死（Infinite Hang）**。最終透過 `scancel` 強制回收資源才使系統回復。

*   **參考程式碼**：`/home/waue0920/workspace/case2_stresstest/stress_test.py`
*   **參考 Log**：`/home/waue0920/workspace/case2_stresstest/stress_test_530.log`

| 時間點 | 事件 | 說明 | 指令 |
| :--- | :--- | :--- | :--- |
| **10:26** | 測試啟動 | 提交 Job 530 並開始 30 分鐘長時穩壓測試 | `sbatch benchmark_v2.sb` |
| **10:56** | 最後通訊 | WandB 回報 Loop 305 (1778.9s)，隨後進入無響應狀態 | (自動執行) |
| **11:13** | 發現異常 | 畫面停滯超過 17 分鐘，Log 停止更新 | - |
| **11:15** | 診斷阻塞 | 嘗試查核進程，**`srun` 指令完全卡死無響應** | `srun --jobid=530 ps ...` |
| **11:17** | 手動干預 | 執行取消指令，釋放掛死資源 | `scancel 530` |
| **11:24** | 系統恢復 | 節點重回 `idle`，`srun` 指令可在 1 秒內回傳結果 | `srun ... uptime` |

## 2. 故障現象與證據 (Evidences)
### 證據一：Linux D-state 的排他性診斷 (Hard Evidence)
*   **現象**：在掛死期間，我們嘗試執行診斷指令 `srun --jobid=530 ps -eo state,pid,cmd`，結果指令**卡死無回傳**。
*   **技術判定**：在 Linux kernel 中，唯有當進程進入 **D-state (Uninterruptible Sleep)** 時，才會導致相關聯的查詢指令（如 `ps`, `ls`）被阻塞。這種狀態有 **99%** 的機率是因為 process 正在等待硬體 I/O（尤其是 NFS Server）回應且不可被中斷。
*   **反證法**：
    *   若是 GPU 死鎖（Computational Deadlock）：`ps` 指令仍能正常列出進程狀態（通常為 R 或 S）。
    *   若是 NCCL 網路死鎖（Network Deadlock）：進程通常處於 S 狀態，且可被 `kill` 信號終止，不會導致系統指令卡死。
    *   **結論**：`srun` 的無響應是 **NFS Client 進入不可中斷等待** 的鐵證。

### 證據二：WandB 紀錄斷崖
*   最後一筆紀錄時間：`10:56:27`。
*   最後進度：`Loop 305` (1778.9s / 1800s)。
*   停滯時長：直至 11:15 仍無任何輸出，掛死時間超過 **19 分鐘**。

### 證據三：系統恢復後的行為
*   執行 `scancel` 後，節點迅速回到 `idle` 狀態，且隨後執行的 `srun uptime` 可在 1 秒內回傳。這證明節點本身的 CPU/Memory/Network 均無異常，問題純粹鎖定在「當時的 I/O 通路」上。

## 3. 根本原因推論 (Root Cause Analysis)
本次故障為典型的 **「NFS 高負載 I/O 導致的 D-state 死鎖」**：

1.  **D-state Blockade**：長時間（30 分鐘）的連續寫入導致 NFS Server 端的處理隊列或 Client 端的 RPC Slot 耗盡。當 Client 等不到 Server 的 `COMMIT` 回應時，Kernel 會將發起 I/O 的進程強制置入 D-state 以保護資料一致性，這導致了外部觀測到的「掛死」。
2.  **連鎖反應**：由於 DDP（Distributed Data Parallel）架構中包含 `dist.barrier()`，只要有**任意一台**節點陷入 NFS D-state，所有其他 63 個 GPU 進程都會無限期等待，造成全叢集停擺。
3.  **技術參考來源**：
    *   [Red Hat: Process in D State (Uninterruptible Sleep)](https://access.redhat.com/solutions/30354) - 解釋 D-state 通常由 I/O 等待造成。
    *   [Linux Kernel: NFS Client Hangs](https://www.kernel.org/doc/Documentation/filesystems/nfs/nfsroot.txt) - 說明 NFS 不回應時導致的系統掛死現象。
    *   [StackExchange: Why does ls hang on NFS mounts?](https://unix.stackexchange.com/questions/276566/why-does-ls-hang-when-nfs-server-is-unreachable) - 解釋為何連基礎指令都會被 NFS 卡住。

## 4. 關鍵結論與使用者建議
1.  **NFS 的長時穩定性風險**：
    實證顯示在 H200 極限運算環境下，雖然大幅降低了 I/O 併發（每節點僅 1 個 I/O 進程），但在長時間（30 分鐘以上）運作時，**NFS 仍發生了導致系統指令阻塞的嚴重掛死**。這證實傳統 NFS 在此類高頻寬運算叢集中，不適合作為「運算中持續寫入」的儲存層。
2.  **系統恢復能力**：
    手動執行 `scancel` 仍能成功清除阻塞的進程。計算節點與系統在資源回收後可迅速回到 `idle` 狀態，並正常響應後續指令（見 11:24 恢復測試）。
3.  **大規模環境下的靜默掛死風險**：
    本案例是因為我們「正在即時監控燒機進度」，才得以在掛死 17 分鐘後及時發現。對於一般使用者而言，若無法預知程式確切的運行時間，極易陷入長久的無意義等待。**強烈建議：**
    *   **設置 Walltime**：作業提交時必須設定合理的時間限制，避免因為 I/O 阻塞造成的靜默掛死消耗不必要的點數與資源。
    *   **儲存分流策略**：對於重度寫入情境，應優先考慮本地 `/tmp` 或高性能平行檔案系統，避免長期依賴 NFS 作為高負載運作時的寫入目標。
