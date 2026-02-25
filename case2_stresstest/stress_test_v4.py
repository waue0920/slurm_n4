import torch
import torch.distributed as dist
import time
import os
import pynvml
import wandb
import numpy as np
import argparse
import csv
import socket
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Stress Test Script")
    parser.add_argument("--project", type=str, default="gpu_stress_test", help="WandB project name")
    parser.add_argument("--duration", type=int, default=1800, help="Test duration in seconds (default: 1800)")
    parser.add_argument("--target_gb", type=int, default=120, help="VRAM to fill per GPU in GB")
    parser.add_argument("--gemm_size", type=int, default=16384, help="Matrix size for GEMM test")
    parser.add_argument("--net_size_mb", type=int, default=1024, help="Data size for NCCL test in MB")
    parser.add_argument("--report_interval_loops", type=int, default=10, help="Interval in loops to report interim results (default: 10, approx 1 min)")
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
        if os.path.exists(local_fn): os.remove(local_fn)

    # 2. NFS 測試 (限 Local Rank 0)
    home_w, home_r = 0.0, 0.0
    work_w, work_r = 0.0, 0.0
    
    if local_rank == 0:
        home_fn = os.path.join(home_dir, f"stress_home_{rank}_{pid}.tmp")
        try:
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
        elif "gb/s" in k.lower() or "bandwidth" in k.lower(): unit = " GB/s"
        elif "mb/s" in k.lower() or "throughput" in k.lower(): unit = " MB/s"
        elif "temp" in k.lower(): unit = " C"
        elif "power" in k.lower(): unit = " W"
        elif "util" in k.lower(): unit = " %"
        
        if isinstance(v, str):
            print(f"{k:<30} | {v:>12}")
        else:
            print(f"{k:<30} | {v:>12.2f}{unit}")
    print("="*55 + "\n")

def print_per_gpu_tflops(rank_tflops, world_size, rank_info=None):
    """
    印出單張 GPU 的效能，幫助識別異常節點/GPU
    """
    print("\n" + "="*55)
    print(f"           INDIVIDUAL GPU COMPUTE (TFLOPS)")
    print("="*55)
    if rank_info:
        header = f"{'Rank':<5} | {'Hostname':<20} | {'TFLOPS':<15}"
        sep = "-" * 5 + "-+-" + "-" * 20 + "-+-" + "-" * 15
    else:
        header = f"{'Rank':<5} | {'TFLOPS':<20}"
        sep = "-" * 5 + "-+-" + "-" * 20
        
    print(header)
    print(sep)
    
    if world_size > 16:
        print(f"Min: {np.min(rank_tflops):.2f} | Max: {np.max(rank_tflops):.2f} | Mean: {np.mean(rank_tflops):.2f}")
        worst_indices = np.argsort(rank_tflops)[:5]
        print(f"Lowest 5 GPUs:")
        for idx in worst_indices:
            host_str = f"({rank_info[idx][0]})" if rank_info else ""
            print(f"  Rank {idx:2d} {host_str:20}: {rank_tflops[idx]:.2f} TFLOPS")
    else:
        for i, val in enumerate(rank_tflops):
            if rank_info:
                host, l_rank = rank_info[i]
                label = f"{host}[G{l_rank}]"
                print(f"{i:<5} | {label:<20} | {val:>10.2f} TFLOPS")
            else:
                print(f"{i:<5} | {val:>20.2f} TFLOPS")
    print("="*55 + "\n")

def save_to_csv(metrics, filename="results.csv"):
    file_exists = os.path.isfile(filename)
    
    # 定義固定的欄位順序 (Header)，避免動態 Hostname 導致 header 變動
    fieldnames = [
        'timestamp', 'hostname', 'GPU Compute (TFLOPS)', 'Network Bandwidth (GB/s)',
        'SSD Write (MB/s)', 'SSD Read (MB/s)', 'Home NFS Write (MB/s)', 'Home NFS Read (MB/s)',
        'Work NFS Write (MB/s)', 'Work NFS Read (MB/s)', 'Avg Temp (C)', 'Avg Power (W)',
        'Avg Util (%)', 'Avg VRAM (MB)', 'Progress'
    ]
    # 固定 GPU0 ~ GPU7 欄位
    for i in range(8):
        fieldnames.append(f"GPU{i}_TFLOPS")
    
    # 預留擴充欄位
    for k in metrics.keys():
        if k not in fieldnames and k != 'timestamp':
            fieldnames.append(k)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        
        # 確保有 timestamp
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        writer.writerow(metrics)

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

    # 收集所有 Rank 的 Hostname 與 local_rank 以便識別是哪張卡
    # all_gather_object 需在 set_device 之後，且需有支援 CPU group 的 backend
    # 這裡使用 gloo 可以不依賴 GPU，透過建立獨立 CPU group 來收集字串資料
    cpu_group = dist.new_group(backend="gloo")
    rank_info = [None for _ in range(world_size)]
    dist.all_gather_object(rank_info, (socket.gethostname(), local_rank), group=cpu_group)

    if rank == 0:
        print(f"Starting Stress Test v4: {world_size} GPUs")
        print(f"Hostname: {socket.gethostname()}")
        print(f"Config: Duration={args.duration}s, VRAM={args.target_gb}GB, GEMM={args.gemm_size}, Net={args.net_size_mb}MB")
        wandb.init(project=args.project, name=f"stress-v4-{datetime.now().strftime('%m%d-%H%M')}")
        wandb.config.update(vars(args))

    loop_count = 0
    stop_signal = torch.tensor([0], dtype=torch.int).to(device=torch.device('cuda', local_rank))
    
    while True:
        loop_count += 1
        
        if rank == 0:
            if time.time() - start_time >= args.duration:
                stop_signal[0] = 1
        dist.broadcast(stop_signal, src=0)
        
        if stop_signal[0] == 1:
            break
            
        # 1. 執行運算並立即取樣 GPU 狀態
        tflops = test_gpu_efficiency(args.gemm_size)
        temp1, pwr1, util1, mem1 = get_gpu_stats()
        
        # 2. 執行通訊並再次取樣
        net_bw = test_network_bw(args.net_size_mb)
        temp2, pwr2, util2, mem2 = get_gpu_stats()
        
        # 3. 執行磁碟 I/O
        ssd_w, ssd_r, home_w, home_r, work_w, work_r = test_disk_io(args.home_dir, args.work_dir)
        
        temp, pwr, util, mem = max(temp1, temp2), max(pwr1, pwr2), max(util1, util2), max(mem1, mem2)
        
        current_vals = [tflops, net_bw, ssd_w, ssd_r, home_w, home_r, work_w, work_r, temp, pwr, util, mem]
        for k, v in zip(history_keys, current_vals):
            history[k].append(v)
        
        # Global aggregation for the current loop
        current_metrics_tensor = torch.tensor(current_vals, device='cuda')
        dist.all_reduce(current_metrics_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            num_nodes = max(1, world_size // local_world_size)
            wandb.log({
                "loop": loop_count,
                "iter_gpu_tflops": current_metrics_tensor[0].item() / world_size,
                "iter_network_gbps": current_metrics_tensor[1].item() / world_size,
                "iter_ssd_w_mbps": current_metrics_tensor[2].item() / world_size,
                "iter_ssd_r_mbps": current_metrics_tensor[3].item() / world_size,
                "iter_home_nfs_w_mbps": current_metrics_tensor[4].item() / num_nodes,
                "iter_home_nfs_r_mbps": current_metrics_tensor[5].item() / num_nodes,
                "iter_work_nfs_w_mbps": current_metrics_tensor[6].item() / num_nodes,
                "iter_work_nfs_r_mbps": current_metrics_tensor[7].item() / num_nodes,
                "iter_gpu_util": current_metrics_tensor[10].item() / world_size,
                "iter_vram_used": current_metrics_tensor[11].item() / world_size,
            })
            if loop_count % 5 == 0:
                print(f"Loop {loop_count} completed.")

        # Heartbeat: Interim Report
        if loop_count % args.report_interval_loops == 0:
            local_avgs = [sum(history[k])/len(history[k]) for k in history_keys]
            local_avgs_tensor = torch.tensor(local_avgs, device='cuda')
            
            # 準備收集所有 Rank 的 TFLOPS
            all_rank_tflops = [torch.zeros(1, device='cuda') for _ in range(world_size)]
            dist.all_gather(all_rank_tflops, local_avgs_tensor[0:1])
            
            # 全域平均
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
                    "Home NFS Read (MB/s)": (global_avgs[5] * world_size / num_nodes).item(),
                    "Work NFS Write (MB/s)": (global_avgs[6] * world_size / num_nodes).item(),
                    "Work NFS Read (MB/s)": (global_avgs[7] * world_size / num_nodes).item(),
                    "Avg Temp (C)": global_avgs[8].item(),
                    "Avg Power (W)": global_avgs[9].item(),
                    "Avg Util (%)": global_avgs[10].item(),
                    "Avg VRAM (MB)": global_avgs[11].item(),
                    "Progress": f"{loop_count}/{loop_count} loops" # 這裡不設估計，直接顯示目前 loop
                }
                print_result_table(interim_metrics, title=f"INTERIM RESULTS (Loop {loop_count})")
                
                # 印出個別 GPU 效能 (Terminal)
                rank_tflops_list = [t.item() for t in all_rank_tflops]
                print_per_gpu_tflops(rank_tflops_list, world_size, rank_info=rank_info)
                
                # 依主機分組效能數據，準備寫入 CSV (Row-per-node)
                node_data = {}
                for i, val in enumerate(rank_tflops_list):
                    host, l_rank = rank_info[i]
                    if host not in node_data:
                        node_data[host] = {}
                    node_data[host][f"GPU{l_rank}_TFLOPS"] = val
                
                # 每個節點寫入一行
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for host, gpus in node_data.items():
                    row = interim_metrics.copy()
                    row['timestamp'] = ts
                    row['hostname'] = host
                    row.update(gpus)
                    save_to_csv(row, filename="results.csv")
                
                # WandB 只需要紀錄一次 (叢集平均與個別數據，個別數據在 WandB 使用動態標籤)
                wb_log = interim_metrics.copy()
                for i, val in enumerate(rank_tflops_list):
                    host, l_rank = rank_info[i]
                    wb_log[f"iter_{host}_G{l_rank}_tflops"] = val
                wandb.log(wb_log)

    # Final Report
    local_final_avgs = [sum(history[k])/len(history[k]) for k in history_keys]
    local_final_tensor = torch.tensor(local_final_avgs, device='cuda')
    
    all_rank_final_tflops = [torch.zeros(1, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_rank_final_tflops, local_final_tensor[0:1])
    
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
            "Home NFS Read (MB/s)": (global_final_avgs[5] * world_size / num_nodes).item(),
            "Work NFS Write (MB/s)": (global_final_avgs[6] * world_size / num_nodes).item(),
            "Work NFS Read (MB/s)": (global_final_avgs[7] * world_size / num_nodes).item(),
            "Avg Temp (C)": global_final_avgs[8].item(),
            "Avg Power (W)": global_final_avgs[9].item(),
            "Avg Util (%)": global_final_avgs[10].item(),
            "Avg VRAM (MB)": global_final_avgs[11].item()
        }
        print_result_table(final_metrics, title="FINAL RESULTS")
        
        rank_final_tflops_list = [t.item() for t in all_rank_final_tflops]
        print_per_gpu_tflops(rank_final_tflops_list, world_size, rank_info=rank_info)
        
        # 依主機分組寫入最終報告
        ts_final = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node_final_data = {}
        for i, val in enumerate(rank_final_tflops_list):
            host, l_rank = rank_info[i]
            if host not in node_final_data:
                node_final_data[host] = {}
            node_final_data[host][f"GPU{l_rank}_TFLOPS"] = val

        for host, gpus in node_final_data.items():
            final_row = final_metrics.copy()
            final_row['timestamp'] = ts_final
            final_row['hostname'] = host
            final_row.update(gpus)
            save_to_csv(final_row, filename="results_20260225.csv")
        
        # WandB Summary 維持紀錄平均指標
        for k, v in final_metrics.items(): 
            wandb.run.summary[k] = v
        wandb.finish()

    pynvml.nvmlShutdown()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
