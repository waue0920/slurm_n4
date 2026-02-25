import torch
import torch.distributed as dist
import time
import os
import pynvml
import wandb
import numpy as np
import argparse
import csv
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Stress Test Script")
    parser.add_argument("--project", type=str, default="gpu_stress_test", help="WandB project name")
    parser.add_argument("--duration", type=int, default=1800, help="Test duration in seconds (default: 1800)")
    parser.add_argument("--target_gb", type=int, default=120, help="VRAM to fill per GPU in GB")
    parser.add_argument("--gemm_size", type=int, default=16384, help="Matrix size for GEMM test")
    parser.add_argument("--net_size_mb", type=int, default=1024, help="Data size for NCCL test in MB")
    parser.add_argument("--report_interval_loops", type=int, default=10, help="Interval in loops to report interim results (default: 10, approx 1 min)")
    # 新增路徑參數，避免寫死使用者帳號
    parser.add_argument("--home_dir", type=str, default=os.path.expanduser("~"), help="Directory for Home NFS test")
    default_work = os.path.join("/work", os.environ.get("USER", "waue0920"))
    parser.add_argument("--work_dir", type=str, default=default_work, help="Directory for Work NFS test")
    return parser.parse_args()

def get_gpu_stats():
    """
    使用 NVML 獲取詳細 GPU 狀態：溫度、功耗、使用率與顯存占用
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / (1024 * 1024)
        return float(temp), float(power), float(util), float(mem_used)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def stress_gpu_memory(target_gb):
    """
    填充 GPU 顯存以增加壓力
    """
    if target_gb <= 0:
        return None
    try:
        # Float16 佔 2 bytes
        num_elements = int((target_gb * 1024 * 1024 * 1024) / 2)
        filler = torch.zeros(num_elements, dtype=torch.float16, device='cuda')
        return filler
    except Exception as e:
        print(f"VRAM Allocation of {target_gb}GB failed: {e}")
        return None

def test_gpu_efficiency(size):
    """
    矩陣乘法 (GEMM) 測試
    """
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)
    iters = 20
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    ops = 2 * (size ** 3) * iters
    tflops = (ops / (end - start)) / 1e12
    return tflops

def test_network_bw(size_mb):
    """
    All-reduce 通訊測試
    """
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, device='cuda')
    iters = 10
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()
    bw = (num_elements * 4 * iters) / (end - start) / 1e9 
    return bw

def test_disk_io(home_dir, work_dir):
    """
    磁碟 I/O 壓力測試函數
    """
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tensor = torch.randn(1024, 1024, 128) # ~512MB
    
    # 加入 PID 以防止多個作業同時執行時，發生檔名衝突 (例如兩個作業都有 Rank 0)
    pid = os.getpid()

    # 1. 本地 SSD
    local_tmp_dir = "/tmp/stress_test"
    local_fn = os.path.join(local_tmp_dir, f"local_rank_{rank}_{pid}.tmp")
    ssd_w, ssd_r = 0.0, 0.0
    
    try:
        os.makedirs(local_tmp_dir, exist_ok=True)
        time.sleep(local_rank * 0.05)
        
        s = time.time()
        torch.save(tensor, local_fn)
        ssd_w = (tensor.nelement() * 4) / (time.time() - s) / 1e6
        s = time.time()
        _ = torch.load(local_fn)
        ssd_r = (tensor.nelement() * 4) / (time.time() - s) / 1e6
    except Exception:
        pass
    finally:
        # 確保發生錯誤也能清理檔案
        if os.path.exists(local_fn): os.remove(local_fn)

    # 2. NFS 測試 (限 Local Rank 0)
    home_w, home_r = 0.0, 0.0
    work_w, work_r = 0.0, 0.0
    
    if local_rank == 0:
        # 使用參數傳入的路徑，加上 PID 避免衝突
        home_fn = os.path.join(home_dir, f"stress_home_{rank}_{pid}.tmp")
        try:
            # 確保目錄存在
            os.makedirs(home_dir, exist_ok=True)
            s = time.time()
            torch.save(tensor, home_fn)
            home_w = (tensor.nelement() * 4) / (time.time() - s) / 1e6
            s = time.time()
            _ = torch.load(home_fn)
            home_r = (tensor.nelement() * 4) / (time.time() - s) / 1e6
        except Exception: 
            pass
        finally:
            if os.path.exists(home_fn): os.remove(home_fn)

        # 使用參數傳入的路徑，加上 PID 避免衝突
        work_fn = os.path.join(work_dir, f"stress_work_{rank}_{pid}.tmp")
        try:
            os.makedirs(work_dir, exist_ok=True)
            s = time.time()
            torch.save(tensor, work_fn)
            work_w = (tensor.nelement() * 4) / (time.time() - s) / 1e6
            s = time.time()
            _ = torch.load(work_fn)
            work_r = (tensor.nelement() * 4) / (time.time() - s) / 1e6
        except Exception: 
            pass
        finally:
            if os.path.exists(work_fn): os.remove(work_fn)
            
    return ssd_w, ssd_r, home_w, home_r, work_w, work_r

def print_result_table(metrics, title="STRESS TEST RESULTS"):
    header = f"{'Metric':<30} | {'Average Value':<20}"
    separator = "-" * 30 + "-+-" + "-" * 20
    print("\n" + "="*55)
    print(f"           {title}")
    print("="*55)
    print(header)
    print(separator)
    for k, v in metrics.items():
        unit = ""
        if "tflops" in k.lower() or "compute" in k.lower(): unit = " TFLOPS"
        elif "gbps" in k.lower() or "bandwidth" in k.lower(): unit = " GB/s"
        elif "mbps" in k.lower() or "throughput" in k.lower(): unit = " MB/s"
        elif "temp" in k.lower(): unit = " C"
        elif "power" in k.lower(): unit = " W"
        elif "util" in k.lower(): unit = " %"
        
        if isinstance(v, str):
            print(f"{k:<30} | {v:>12}")
        else:
            print(f"{k:<30} | {v:>12.2f}{unit}")
    print("="*55 + "\n")

def save_to_csv(metrics, filename="results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp'] + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        row.update(metrics)
        writer.writerow(row)

def main():
    args = parse_args()
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    
    # 動態取得 local_world_size，確保 NFS 吞吐量計算正確
    local_world_size = torch.cuda.device_count()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    pynvml.nvmlInit()
    _vram_filler = stress_gpu_memory(target_gb=args.target_gb)
    
    start_time = time.time()
    history_keys = ["tflops", "net_bw", "ssd_w", "ssd_r", "home_w", "home_r", "work_w", "work_r", "temp", "power", "util", "mem"]
    history = {k: [] for k in history_keys}

    if rank == 0:
        print(f"Starting Stress Test: {world_size} GPUs")
        print(f"Config: Duration={args.duration}s, VRAM={args.target_gb}GB, GEMM={args.gemm_size}, Net={args.net_size_mb}MB")
        wandb.init(project=args.project, name=f"stress-{datetime.now().strftime('%m%d-%H%M')}")
        wandb.config.update(vars(args))

    loop_count = 0
    # 採用統一廣播終止信號，防止各節點因時鐘微小偏差導致的「收尾死鎖」
    stop_signal = torch.tensor([0], dtype=torch.int).to(device=torch.device('cuda', local_rank))
    
    while True:
        loop_count += 1
        
        # 由 Rank 0 檢查時間並廣播狀態
        if rank == 0:
            if time.time() - start_time >= args.duration:
                stop_signal[0] = 1
        dist.broadcast(stop_signal, src=0)
        
        if stop_signal[0] == 1:
            break
            
        # 1. 執行運算並立即取樣 GPU 狀態 (確保抓到滿載數據)
        tflops = test_gpu_efficiency(args.gemm_size)
        # 取樣第一次 GPU 狀態 (針對運算壓力)
        temp1, pwr1, util1, mem1 = get_gpu_stats()
        
        # 2. 執行通訊並再次取樣
        net_bw = test_network_bw(args.net_size_mb)
        # 取樣第二次 GPU 狀態 (針對通訊壓力)
        temp2, pwr2, util2, mem2 = get_gpu_stats()
        
        # 3. 執行磁碟 I/O (通常較慢，會導致 GPU 降溫)
        # 傳入參數中的路徑
        ssd_w, ssd_r, home_w, home_r, work_w, work_r = test_disk_io(args.home_dir, args.work_dir)
        
        # 綜合彙整本次迴圈數據
        # 取兩次負載取樣的最大值，以反映真實壓測強度
        temp, pwr, util, mem = max(temp1, temp2), max(pwr1, pwr2), max(util1, util2), max(mem1, mem2)
        
        # 記錄本次數據
        current_vals = [tflops, net_bw, ssd_w, ssd_r, home_w, home_r, work_w, work_r, temp, pwr, util, mem]
        for k, v in zip(history_keys, current_vals):
            history[k].append(v)
        
        # Global aggregation for the current loop (used by WandB)
        current_metrics_tensor = torch.tensor(current_vals, device='cuda')
        dist.all_reduce(current_metrics_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            # 計算節點數，用於 NFS 平均值還原
            num_nodes = max(1, world_size // local_world_size)
            wandb.log({
                "loop": loop_count,
                "iter_gpu_tflops": current_metrics_tensor[0].item() / world_size,
                "iter_network_gbps": current_metrics_tensor[1].item() / world_size,
                "iter_ssd_w_mbps": current_metrics_tensor[2].item() / world_size,
                "iter_home_nfs_w_mbps": current_metrics_tensor[4].item() / num_nodes,
                "iter_work_nfs_w_mbps": current_metrics_tensor[6].item() / num_nodes,
                "iter_gpu_util": current_metrics_tensor[10].item() / world_size,
                "iter_vram_used": current_metrics_tensor[11].item() / world_size,
            })
            if loop_count % 5 == 0:
                print(f"Loop {loop_count} completed.")

        # Heartbeat: Interim Report every N loops
        if loop_count % args.report_interval_loops == 0:
            # Calculate global averages of history so far
            local_avgs = [sum(history[k])/len(history[k]) for k in history_keys]
            local_avgs_tensor = torch.tensor(local_avgs, device='cuda')
            dist.all_reduce(local_avgs_tensor, op=dist.ReduceOp.SUM)
            global_avgs = local_avgs_tensor / world_size
            
            if rank == 0:
                num_nodes = max(1, world_size // local_world_size)
                interim_metrics = {
                    "GPU Compute (TFLOPS)": global_avgs[0].item(),
                    "Network Bandwidth (GB/s)": global_avgs[1].item(),
                    "SSD Write (MB/s)": global_avgs[2].item(),
                    "SSD Read (MB/s)": global_avgs[3].item(),
                    "Home NFS Write (MB/s)": (global_avgs[4] * world_size / num_nodes).item(),
                    "Work NFS Write (MB/s)": (global_avgs[6] * world_size / num_nodes).item(),
                    "Avg Temp (C)": global_avgs[8].item(),
                    "Avg Power (W)": global_avgs[9].item(),
                    "Avg Util (%)": global_avgs[10].item(),
                    "Avg VRAM (MB)": global_avgs[11].item(),
                    "Progress": f"{loop_count}/{args.duration//6} loops (est)"
                }
                print_result_table(interim_metrics, title=f"INTERIM RESULTS (Loop {loop_count})")
                save_to_csv(interim_metrics, filename="results.csv")

    # Final Report
    local_final_avgs = [sum(history[k])/len(history[k]) for k in history_keys]
    local_final_tensor = torch.tensor(local_final_avgs, device='cuda')
    dist.all_reduce(local_final_tensor, op=dist.ReduceOp.SUM)
    global_final_avgs = local_final_tensor / world_size
    
    if rank == 0:
        num_nodes = max(1, world_size // local_world_size)
        final_metrics = {
            "GPU Compute (TFLOPS)": global_final_avgs[0].item(),
            "Network Bandwidth (GB/s)": global_final_avgs[1].item(),
            "SSD Write (MB/s)": global_final_avgs[2].item(),
            "SSD Read (MB/s)": global_final_avgs[3].item(),
            "Home NFS Write (MB/s)": (global_final_avgs[4] * world_size / num_nodes).item(),
            "Work NFS Write (MB/s)": (global_final_avgs[6] * world_size / num_nodes).item(),
            "Avg Temp (C)": global_final_avgs[8].item(),
            "Avg Power (W)": global_final_avgs[9].item(),
            "Avg Util (%)": global_final_avgs[10].item(),
            "Avg VRAM (MB)": global_final_avgs[11].item()
        }
        print_result_table(final_metrics, title="FINAL RESULTS")
        save_to_csv(final_metrics, filename="results.csv")
        for k, v in final_metrics.items(): wandb.run.summary[k] = v
        wandb.finish()

    pynvml.nvmlShutdown()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
