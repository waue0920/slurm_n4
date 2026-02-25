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
import mmap
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

def test_direct_io(filename, size_mb=512):
    """
    使用 O_DIRECT 進行原始磁碟 I/O 測試，避開 OS Cache
    """
    size = size_mb * 1024 * 1024
    # O_DIRECT 需要記憶體對齊。mmap 通常會回傳對齊 page 的記憶體。
    buf = mmap.mmap(-1, size)
    # 填充一些數據避免寫入全零（有些檔案系統會優化）
    buf[:1024] = b"a" * 1024 
    
    try:
        # 開啟檔案並設定 O_DIRECT
        # 注意：某些檔案系統（如 tmpfs）不支援 O_DIRECT，若失敗會報錯
        fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT)
        
        # Write
        start = time.time()
        os.write(fd, buf)
        os.fsync(fd) 
        w_time = time.time() - start
        w_speed = size_mb / w_time
        
        # Read
        os.lseek(fd, 0, 0)
        start = time.time()
        os.read(fd, size)
        r_time = time.time() - start
        r_speed = size_mb / r_time
        
        os.close(fd)
        return w_speed, r_speed
    except Exception as e:
        # 如果不支援 O_DIRECT，嘗試一般 IO 但手動同步
        try:
            fd = os.open(filename, os.O_CREAT | os.O_RDWR)
            start = time.time()
            os.write(fd, buf)
            os.fsync(fd)
            w_speed = size_mb / (time.time() - start)
            
            # 清除快取 (POSIX 提示)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            
            os.lseek(fd, 0, 0)
            start = time.time()
            os.read(fd, size)
            r_speed = size_mb / (time.time() - start)
            os.close(fd)
            return w_speed, r_speed
        except:
            return 0.0, 0.0
    finally:
        if os.path.exists(filename):
            try: os.remove(filename)
            except: pass

def test_disk_io(home_dir, work_dir):
    """
    磁碟 I/O 壓力測試函數
    修正：限定單一程序執行，並加入 Direct IO 與強制同步 (fsync)
    """
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # 測試資料大小
    mb_size = 512
    tensor = torch.randn(1024, 1024, mb_size // 4) # ~512MB
    tensor_bytes = tensor.nelement() * 4
    
    pid = os.getpid()
    
    ssd_w, ssd_r = 0.0, 0.0
    ssd_direct_w, ssd_direct_r = 0.0, 0.0
    home_w, home_r = 0.0, 0.0
    home_direct_w, home_direct_r = 0.0, 0.0
    work_w, work_r = 0.0, 0.0
    work_direct_w, work_direct_r = 0.0, 0.0

    # 1. 本地 SSD (每節點僅由 Local Rank 0 執行)
    if local_rank == 0:
        local_tmp_dir = "/tmp/stress_test"
        os.makedirs(local_tmp_dir, exist_ok=True)
        
        # Standard Torch IO with Cache Invalidation
        local_fn = os.path.join(local_tmp_dir, f"torch_io_{rank}_{pid}.tmp")
        try:
            # Write + fsync
            s = time.time()
            torch.save(tensor, local_fn)
            fd = os.open(local_fn, os.O_WRONLY)
            os.fsync(fd)
            os.close(fd)
            ssd_w = tensor_bytes / (time.time() - s) / 1e6
            
            # Invalidate Cache
            fd = os.open(local_fn, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            
            # Read
            s = time.time()
            _ = torch.load(local_fn)
            ssd_r = tensor_bytes / (time.time() - s) / 1e6
        except Exception: pass
        finally:
            if os.path.exists(local_fn): os.remove(local_fn)
            
        # Direct IO (O_DIRECT)
        direct_fn = os.path.join(local_tmp_dir, f"direct_io_{rank}_{pid}.tmp")
        ssd_direct_w, ssd_direct_r = test_direct_io(direct_fn, size_mb=mb_size)

    # 2. NFS 測試 (全域僅由 Rank 0 執行)
    if rank == 0:
        # Home NFS
        home_fn = os.path.join(home_dir, f"stress_home_{rank}_{pid}.tmp")
        try:
            os.makedirs(home_dir, exist_ok=True)
            # Standard IO + fsync
            s = time.time()
            torch.save(tensor, home_fn)
            fd = os.open(home_fn, os.O_WRONLY)
            os.fsync(fd) # 重要：強制將緩衝推向網路與伺服器硬碟
            os.close(fd)
            home_w = tensor_bytes / (time.time() - s) / 1e6
            
            # Invalidate Cache
            fd = os.open(home_fn, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            
            s = time.time()
            _ = torch.load(home_fn)
            home_r = tensor_bytes / (time.time() - s) / 1e6
            
            # Direct IO on NFS (might fail on some mounts, but we try)
            h_dir_fn = os.path.join(home_dir, f"direct_home_{rank}_{pid}.tmp")
            home_direct_w, home_direct_r = test_direct_io(h_dir_fn, size_mb=mb_size)
        except Exception: pass
        finally:
            if os.path.exists(home_fn): os.remove(home_fn)

        # Work NFS
        work_fn = os.path.join(work_dir, f"stress_work_{rank}_{pid}.tmp")
        try:
            os.makedirs(work_dir, exist_ok=True)
            # Standard IO + fsync
            s = time.time()
            torch.save(tensor, work_fn)
            fd = os.open(work_fn, os.O_WRONLY)
            os.fsync(fd) # 重要：徹底測試網路與磁碟實際寫入
            os.close(fd)
            work_w = tensor_bytes / (time.time() - s) / 1e6
            
            # Invalidate Cache
            fd = os.open(work_fn, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            
            s = time.time()
            _ = torch.load(work_fn)
            work_r = tensor_bytes / (time.time() - s) / 1e6

            # Direct IO on Work NFS
            w_dir_fn = os.path.join(work_dir, f"direct_work_{rank}_{pid}.tmp")
            work_direct_w, work_direct_r = test_direct_io(w_dir_fn, size_mb=mb_size)
        except Exception: pass
        finally:
            if os.path.exists(work_fn): os.remove(work_fn)
            
    return ssd_w, ssd_r, ssd_direct_w, ssd_direct_r, home_w, home_r, home_direct_w, home_direct_r, work_w, work_r, work_direct_w, work_direct_r

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
    
    # 定義固定的欄位順序 (Header)
    fieldnames = [
        'timestamp', 'hostname', 'GPU Compute (TFLOPS)', 'Network Bandwidth (GB/s)',
        'SSD Write (MB/s)', 'SSD Read (MB/s)', 'SSD Direct Write (MB/s)', 'SSD Direct Read (MB/s)',
        'Home NFS Write (MB/s)', 'Home NFS Read (MB/s)', 'Home Direct Write (MB/s)', 'Home Direct Read (MB/s)',
        'Work NFS Write (MB/s)', 'Work NFS Read (MB/s)', 'Work Direct Write (MB/s)', 'Work Direct Read (MB/s)',
        'Avg Temp (C)', 'Avg Power (W)',
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
        
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        rounded = {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
        writer.writerow(rounded)

def main():
    args = parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_gpus = torch.cuda.device_count()
    print(f"[INIT] PID={os.getpid()} LOCAL_RANK={local_rank}, CUDA device_count={num_gpus}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
    local_rank = min(local_rank, num_gpus - 1)
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    local_world_size = torch.cuda.device_count()
    world_size = dist.get_world_size()
    num_nodes = max(1, world_size // local_world_size)
    
    pynvml.nvmlInit()
    _vram_filler = stress_gpu_memory(target_gb=args.target_gb)
    
    start_time = time.time()
    history_keys = [
        "tflops", "net_bw", "ssd_w", "ssd_r", "ssd_direct_w", "ssd_direct_r", 
        "home_w", "home_r", "home_direct_w", "home_direct_r",
        "work_w", "work_r", "work_direct_w", "work_direct_r",
        "temp", "power", "util", "mem"
    ]
    history = {k: [] for k in history_keys}

    rank_info = [None for _ in range(world_size)]
    dist.all_gather_object(rank_info, (socket.gethostname(), local_rank))

    if rank == 0:
        print(f"Starting Stress Test v4 (Strict IO): {world_size} GPUs, {num_nodes} nodes")
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
            
        tflops = test_gpu_efficiency(args.gemm_size)
        temp1, pwr1, util1, mem1 = get_gpu_stats()
        
        net_bw = test_network_bw(args.net_size_mb)
        temp2, pwr2, util2, mem2 = get_gpu_stats()
        
        # 磁碟 I/O
        io_results = test_disk_io(args.home_dir, args.work_dir)
        
        temp, pwr, util, mem = max(temp1, temp2), max(pwr1, pwr2), max(util1, util2), max(mem1, mem2)
        
        current_vals = [tflops, net_bw] + list(io_results) + [temp, pwr, util, mem]
        for k, v in zip(history_keys, current_vals):
            history[k].append(v)
        
        # Global aggregation
        current_metrics_tensor = torch.tensor(current_vals, device='cuda')
        dist.all_reduce(current_metrics_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            wandb.log({
                "loop": loop_count,
                "iter_gpu_tflops": current_metrics_tensor[0].item() / world_size,
                "iter_network_gbps": current_metrics_tensor[1].item() / world_size,
                "iter_ssd_w_mbps": current_metrics_tensor[2].item() / num_nodes,
                "iter_ssd_r_mbps": current_metrics_tensor[3].item() / num_nodes,
                "iter_ssd_direct_w_mbps": current_metrics_tensor[4].item() / num_nodes,
                "iter_ssd_direct_r_mbps": current_metrics_tensor[5].item() / num_nodes,
                "iter_home_nfs_w_mbps": current_metrics_tensor[6].item(), # Rank 0 only
                "iter_home_nfs_r_mbps": current_metrics_tensor[7].item(), # Rank 0 only
                "iter_home_direct_w_mbps": current_metrics_tensor[8].item(), # Rank 0 only
                "iter_home_direct_r_mbps": current_metrics_tensor[9].item(), # Rank 0 only
                "iter_work_nfs_w_mbps": current_metrics_tensor[10].item(), # Rank 0 only
                "iter_work_nfs_r_mbps": current_metrics_tensor[11].item(), # Rank 0 only
                "iter_work_direct_w_mbps": current_metrics_tensor[12].item(), # Rank 0 only
                "iter_work_direct_r_mbps": current_metrics_tensor[13].item(), # Rank 0 only
                "iter_gpu_util": current_metrics_tensor[16].item() / world_size,
                "iter_vram_used": current_metrics_tensor[17].item() / world_size,
            })
            if loop_count % 5 == 0:
                print(f"Loop {loop_count} completed.")

        if loop_count % args.report_interval_loops == 0:
            local_avgs = [sum(history[k])/len(history[k]) for k in history_keys]
            local_avgs_tensor = torch.tensor(local_avgs, device='cuda')
            
            all_rank_tflops = [torch.zeros(1, device='cuda') for _ in range(world_size)]
            dist.all_gather(all_rank_tflops, local_avgs_tensor[0:1])
            
            dist.all_reduce(local_avgs_tensor, op=dist.ReduceOp.SUM)
            global_sum = local_avgs_tensor
            
            if rank == 0:
                interim_metrics = {
                    "GPU Compute (TFLOPS)": global_sum[0].item() / world_size,
                    "Network Bandwidth (GB/s)": global_sum[1].item() / world_size,
                    "SSD Write (MB/s)": global_sum[2].item() / num_nodes,
                    "SSD Read (MB/s)": global_sum[3].item() / num_nodes,
                    "SSD Direct Write (MB/s)": global_sum[4].item() / num_nodes,
                    "SSD Direct Read (MB/s)": global_sum[5].item() / num_nodes,
                    "Home NFS Write (MB/s)": global_sum[6].item(),
                    "Home NFS Read (MB/s)": global_sum[7].item(),
                    "Home Direct Write (MB/s)": global_sum[8].item(),
                    "Home Direct Read (MB/s)": global_sum[9].item(),
                    "Work NFS Write (MB/s)": global_sum[10].item(),
                    "Work NFS Read (MB/s)": global_sum[11].item(),
                    "Work Direct Write (MB/s)": global_sum[12].item(),
                    "Work Direct Read (MB/s)": global_sum[13].item(),
                    "Avg Temp (C)": global_sum[14].item() / world_size,
                    "Avg Power (W)": global_sum[15].item() / world_size,
                    "Avg Util (%)": global_sum[16].item() / world_size,
                    "Avg VRAM (MB)": global_sum[17].item() / world_size,
                    "Progress": f"{loop_count} loops"
                }
                print_result_table(interim_metrics, title=f"INTERIM RESULTS (Loop {loop_count})")
                
                rank_tflops_list = [t.item() for t in all_rank_tflops]
                print_per_gpu_tflops(rank_tflops_list, world_size, rank_info=rank_info)
                
                node_data = {}
                for i, val in enumerate(rank_tflops_list):
                    host, l_rank = rank_info[i]
                    if host not in node_data:
                        node_data[host] = {}
                    node_data[host][f"GPU{l_rank}_TFLOPS"] = val
                
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for host, gpus in node_data.items():
                    row = interim_metrics.copy()
                    row['timestamp'] = ts
                    row['hostname'] = host
                    row.update(gpus)
                    save_to_csv(row, filename="results.csv")
                
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
    global_final_sum = local_final_tensor
    
    if rank == 0:
        final_metrics = {
            "GPU Compute (TFLOPS)": global_final_sum[0].item() / world_size,
            "Network Bandwidth (GB/s)": global_final_sum[1].item() / world_size,
            "SSD Write (MB/s)": global_final_sum[2].item() / num_nodes,
            "SSD Read (MB/s)": global_final_sum[3].item() / num_nodes,
            "SSD Direct Write (MB/s)": global_final_sum[4].item() / num_nodes,
            "SSD Direct Read (MB/s)": global_final_sum[5].item() / num_nodes,
            "Home NFS Write (MB/s)": global_final_sum[6].item(),
            "Home NFS Read (MB/s)": global_final_sum[7].item(),
            "Home Direct Write (MB/s)": global_final_sum[8].item(),
            "Home Direct Read (MB/s)": global_final_sum[9].item(),
            "Work NFS Write (MB/s)": global_final_sum[10].item(),
            "Work NFS Read (MB/s)": global_final_sum[11].item(),
            "Work Direct Write (MB/s)": global_final_sum[12].item(),
            "Work Direct Read (MB/s)": global_final_sum[13].item(),
            "Avg Temp (C)": global_final_sum[14].item() / world_size,
            "Avg Power (W)": global_final_sum[15].item() / world_size,
            "Avg Util (%)": global_final_sum[16].item() / world_size,
            "Avg VRAM (MB)": global_final_sum[17].item() / world_size
        }
        print_result_table(final_metrics, title="FINAL RESULTS")
        
        rank_final_tflops_list = [t.item() for t in all_rank_final_tflops]
        print_per_gpu_tflops(rank_final_tflops_list, world_size, rank_info=rank_info)
        
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
            # 寫入一個帶日期的最終結果檔案
            save_to_csv(final_row, filename=f"results_{datetime.now().strftime('%Y%m%d')}.csv")
        
        for k, v in final_metrics.items(): 
            wandb.run.summary[k] = v
        wandb.finish()

    pynvml.nvmlShutdown()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
