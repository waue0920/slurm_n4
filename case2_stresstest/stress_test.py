import torch
import torch.distributed as dist
import time
import os
import pynvml
import wandb
import numpy as np
from datetime import datetime

def get_gpu_stats():
    """
    使用 NVML 獲取詳細 GPU 狀態：溫度、功耗、使用率與顯存占用
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        
        # 獲取溫度 (C) 與 功耗 (W)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        
        # 獲取 GPU 核心使用率 (%)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        
        # 獲取顯存資訊 (MB)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / (1024 * 1024)
        
        return float(temp), float(power), float(util), float(mem_used)
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

def stress_gpu_memory(target_gb=120):
    """
    填充 GPU 顯存以增加壓力，模擬大型模型載入場景。
    H200 有 141GB，預設填充到 120GB。
    """
    try:
        # Float16 佔 2 bytes
        num_elements = int((target_gb * 1024 * 1024 * 1024) / 2)
        # 分配但不初始化以加快速度，隨後填入隨機值
        filler = torch.zeros(num_elements, dtype=torch.float16, device='cuda')
        # 保持引用以防止垃圾回收
        return filler
    except Exception as e:
        print(f"VRAM Allocation failed: {e}")
        return None

def test_gpu_efficiency():
    """
    矩陣乘法 (GEMM) 測試：衡量 GPU 運算效能 (TFLOPS)
    """
    size = 16384 
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)
    iters = 20
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    # 浮點運算量：2 * M * N * K
    ops = 2 * (size ** 3) * iters
    tflops = (ops / (end - start)) / 1e12
    return tflops

def test_network_bw():
    """
    All-reduce 通訊測試：衡量節點間網路頻寬 (GB/s)
    """
    size = 1024 * 1024 * 256 # 1GB 資料量
    tensor = torch.randn(size, device='cuda')
    iters = 10
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()
    # 輸出頻寬 (GB/s)
    bw = (size * 4 * iters) / (end - start) / 1e9 
    return bw

def test_disk_io():
    """
    磁碟 I/O 壓力測試函數
    
    設計邏輯：
    1. 本地 SSD (/tmp): 每台節點有 8 個 Rank 同時寫入，模擬多 GPU 訓練時同步存取 Checkpoint 的壓力。
       使用 time.sleep 進行毫秒級交錯，防止檔案系統 Metadata 瞬間鎖定。
    2. Home/Work NFS: 採「低併發模式」，每台節點僅由 Local Rank 0 執行。
       全叢集總計僅 8 個進程同時操作，避免造成網路掛載空間（NFS）崩潰。
    """
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # 測試資料量設為約 512MB
    tensor = torch.randn(1024, 1024, 128) 
    
    # --- 1. 本地 SSD 測試 (/tmp) ---
    # 併發數：每節點 8 個 R/W 操作
    local_tmp_dir = "/tmp/stress_test_tmp"
    os.makedirs(local_tmp_dir, exist_ok=True)
    local_fn = os.path.join(local_tmp_dir, f"local_rank_{rank}.tmp")
    
    # 稍微交錯啟動，避免 64 個進程在同一微秒調用 open()
    time.sleep(local_rank * 0.05)
    
    s = time.time()
    torch.save(tensor, local_fn)
    ssd_w = (tensor.nelement() * 4) / (time.time() - s) / 1e6
    
    s = time.time()
    _ = torch.load(local_fn)
    ssd_r = (tensor.nelement() * 4) / (time.time() - s) / 1e6
    
    # 測試完畢清理本地檔案
    if os.path.exists(local_fn): os.remove(local_fn)

    # --- 2. Home NFS & Work NFS 測試 ---
    # 併發數：全叢集僅 8 個 R/W 操作 (每台節點一個進程)
    home_w, home_r = 0.0, 0.0
    work_w, work_r = 0.0, 0.0
    
    if local_rank == 0:
        # Home NFS 測試
        home_fn = f"/home/waue0920/stress_test_home_{rank}.tmp"
        try:
            s = time.time()
            torch.save(tensor, home_fn)
            home_w = (tensor.nelement() * 4) / (time.time() - s) / 1e6
            s = time.time()
            _ = torch.load(home_fn)
            home_r = (tensor.nelement() * 4) / (time.time() - s) / 1e6
            if os.path.exists(home_fn): os.remove(home_fn)
        except Exception as e:
            print(f"Home NFS Error on Rank {rank}: {e}")

        # Work NFS 測試
        work_fn = f"/work/waue0920/stress_test_work_{rank}.tmp"
        try:
            # 確保工作目錄存在
            os.makedirs("/work/waue0920", exist_ok=True)
            s = time.time()
            torch.save(tensor, work_fn)
            work_w = (tensor.nelement() * 4) / (time.time() - s) / 1e6
            s = time.time()
            _ = torch.load(work_fn)
            work_r = (tensor.nelement() * 4) / (time.time() - s) / 1e6
            if os.path.exists(work_fn): os.remove(work_fn)
        except Exception as e:
            print(f"Work NFS Error on Rank {rank}: {e}")
            
    return ssd_w, ssd_r, home_w, home_r, work_w, work_r

def print_result_table(metrics):
    """
    在終端印出格式化效能報告表格
    """
    header = f"{'Metric':<30} | {'Average Value':<20}"
    separator = "-" * 30 + "-+-" + "-" * 20
    print("\n" + "="*55)
    print("         H200 CLUSTER BURN-IN TEST RESULTS")
    print("="*55)
    print(header)
    print(separator)
    for k, v in metrics.items():
        unit = ""
        if "tflops" in k.lower(): unit = " TFLOPS"
        elif "gbps" in k.lower(): unit = " GB/s"
        elif "mbps" in k.lower(): unit = " MB/s"
        elif "temp" in k.lower(): unit = " C"
        elif "power" in k.lower(): unit = " W"
        elif "time" in k.lower(): unit = " s"
        
        print(f"{k:<30} | {v:>12.2f}{unit}")
    print("="*55 + "\n")

def main(project_name):
    # 初始化分散式環境 (torchrun)
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    # 初始化 NVML
    pynvml.nvmlInit()
    
    # --- 新增：GPU 顯存填充 ---
    # 為 H200 預留約 120GB 顯存壓力
    _vram_filler = stress_gpu_memory(target_gb=120)
    
    # 設定測試時長 (30 分鐘 = 1800 秒)
    test_duration = 1800 
    start_time = time.time()
    
    # 數據統計暫存
    history = {
        "tflops": [],
        "net_bw": [],
        "ssd_w": [],
        "ssd_r": [],
        "home_w": [],
        "home_r": [],
        "work_w": [],
        "work_r": [],
        "temp": [],
        "power": []
    }

    if rank == 0:
        print(f"Starting 30-minute burn-in test on {world_size} GPUs...")
        # 直接傳入變數即可，更簡潔
        wandb.init(project=project_name, name=f"burnin-64g-{datetime.now().strftime('%m%d-%H%M')}")

    loop_count = 0
    while time.time() - start_time < test_duration:
        loop_count += 1
        
        # 執行各項壓力測試
        tflops = test_gpu_efficiency()
        net_bw = test_network_bw()
        ssd_w, ssd_r, home_w, home_r, work_w, work_r = test_disk_io()
        temp, pwr, util, mem = get_gpu_stats()
        
        # 記錄本次數據
        history["tflops"].append(tflops)
        history["net_bw"].append(net_bw)
        history["ssd_w"].append(ssd_w)
        history["ssd_r"].append(ssd_r)
        history["home_w"].append(home_w)
        history["home_r"].append(home_r)
        history["work_w"].append(work_w)
        history["work_r"].append(work_r)
        history["temp"].append(temp)
        history["power"].append(pwr)
        # 新增監控紀錄
        history.setdefault("util", []).append(util)
        history.setdefault("mem", []).append(mem)
        
        # 聚合數據
        # 數值：tflops, net_bw, ssd_w, ssd_r, home_w, home_r, work_w, work_r, temp, pwr, util, mem
        current_metrics = torch.tensor([tflops, net_bw, ssd_w, ssd_r, home_w, home_r, work_w, work_r, temp, pwr, util, mem], device='cuda')
        dist.all_reduce(current_metrics, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            elapsed = time.time() - start_time
            num_nodes = world_size // 8
            wandb.log({
                "loop": loop_count,
                "elapsed_time": elapsed,
                "iter_gpu_tflops": current_metrics[0].item() / world_size,
                "iter_network_gbps": current_metrics[1].item() / world_size,
                "iter_ssd_write_mbps": current_metrics[2].item() / world_size,
                "iter_ssd_read_mbps": current_metrics[3].item() / world_size,
                "iter_home_nfs_w_mbps": current_metrics[4].item() / num_nodes,
                "iter_home_nfs_r_mbps": current_metrics[5].item() / num_nodes,
                "iter_work_nfs_w_mbps": current_metrics[6].item() / num_nodes,
                "iter_work_nfs_r_mbps": current_metrics[7].item() / num_nodes,
                "iter_gpu_temp_c": current_metrics[8].item() / world_size,
                "iter_gpu_power_w": current_metrics[9].item() / world_size,
                "iter_gpu_util_pct": current_metrics[10].item() / world_size,
                "iter_vram_used_mb": current_metrics[11].item() / world_size,
            })
            
            if loop_count % 5 == 0:
                print(f"[{elapsed:6.1f}s / {test_duration}s] Loop {loop_count} completed, metrics logged to wandb.")

    # 計算本地平均值
    local_avgs = {k: sum(v)/len(v) if v else 0.0 for k, v in history.items()}
    local_tensor = torch.tensor(list(local_avgs.values()), device='cuda')
    
    # 跨節點聚合所有 GPU 數據 (求總和後平均)
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    global_avgs = local_tensor / world_size
    
    if rank == 0:
        keys = list(local_avgs.keys())
        num_nodes = world_size // 8
        final_metrics = {
            "GPU Compute Performance": global_avgs[keys.index("tflops")].item(),
            "Network (NCCL) Bandwidth": global_avgs[keys.index("net_bw")].item(),
            "SSD Write (Local)": global_avgs[keys.index("ssd_w")].item(),
            "SSD Read (Local)": global_avgs[keys.index("ssd_r")].item(),
            "Home NFS Write": (global_avgs[keys.index("home_w")] * world_size / num_nodes).item(),
            "Home NFS Read": (global_avgs[keys.index("home_r")] * world_size / num_nodes).item(),
            "Work NFS Write": (global_avgs[keys.index("work_w")] * world_size / num_nodes).item(),
            "Work NFS Read": (global_avgs[keys.index("work_r")] * world_size / num_nodes).item(),
            "Average GPU Temperature": global_avgs[keys.index("temp")].item(),
            "Average GPU Power Draw": global_avgs[keys.index("power")].item(),
            "Average GPU Utilization %": global_avgs[keys.index("util") if "util" in keys else 0].item(),
            "Average VRAM Usage (MB)": global_avgs[keys.index("mem") if "mem" in keys else 0].item(),
            "Total Loops Executed": float(loop_count)
        }
        
        # 輸出最終報表
        print_result_table(final_metrics)
        for k, v in final_metrics.items():
            wandb.run.summary[k] = v
        wandb.finish()

    # 關閉 NVML 與 分散式環境
    pynvml.nvmlShutdown()
    dist.destroy_process_group()

if __name__ == "__main__":
    pro_name="n4_stress_benchmark_h200"
    main(pro_name)
