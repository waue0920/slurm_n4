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
import psutil
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Node GPU Stress Test (Local Torchrun)")
    parser.add_argument("--project", type=str, default="openstack_vm_benchmark", help="WandB project name")
    parser.add_argument("--duration", type=int, default=600, help="Test duration in seconds")
    parser.add_argument("--target_gb", type=int, default=80, help="VRAM to fill per GPU in GB")
    parser.add_argument("--gemm_size", type=int, default=16384, help="Matrix size for GEMM test")
    parser.add_argument("--net_size_mb", type=int, default=1024, help="Data size for NCCL test in MB")
    parser.add_argument("--report_interval", type=int, default=5, help="Report every X loops")
    parser.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    return parser.parse_args()

def get_gpu_stats(handle):
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / (1024 * 1024)
        return float(temp), float(power), float(util), float(mem_used)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def stress_gpu_memory(target_gb):
    if target_gb <= 0:
        return None
    try:
        num_elements = int((target_gb * 1024 * 1024 * 1024) / 2) # float16
        filler = torch.zeros(num_elements, dtype=torch.float16, device='cuda')
        print(f"[VRAM] Allocated {target_gb}GB on rank {dist.get_rank()}")
        return filler
    except Exception as e:
        print(f"[VRAM] Allocation failed on rank {dist.get_rank()}: {e}")
        return None

def test_gpu_efficiency(size):
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
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, device='cuda')
    iters = 10
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()
    bw = (num_elements * 4 * iters) / (end - start) / 1e9 # GB/s
    return bw

def test_disk_io():
    """Simple local disk I/O test"""
    mb_size = 512
    tensor = torch.randn(1024, 1024, mb_size // 4)
    tensor_bytes = tensor.nelement() * 4
    
    local_fn = f"/tmp/stress_test_{os.getpid()}.tmp"
    w_speed, r_speed = 0.0, 0.0
    try:
        # Write
        start = time.time()
        torch.save(tensor, local_fn)
        # Force sync
        fd = os.open(local_fn, os.O_WRONLY)
        os.fsync(fd)
        os.close(fd)
        w_speed = tensor_bytes / (time.time() - start) / 1e6
        
        # Read
        start = time.time()
        _ = torch.load(local_fn)
        r_speed = tensor_bytes / (time.time() - start) / 1e6
    except Exception:
        pass
    finally:
        if os.path.exists(local_fn):
            os.remove(local_fn)
            
    return w_speed, r_speed

def save_to_csv(metrics, filename="results.csv"):
    file_exists = os.path.isfile(filename)
    
    fieldnames = [
        'timestamp', 'minute', 'avg_tflops', 'avg_net_bw_gbs',
        'avg_disk_w_mbs', 'avg_disk_r_mbs', 
        'avg_cpu_util', 'avg_ram_used_gb',
        'avg_temp_c', 'avg_power_w', 'avg_gpu_util_pct', 'avg_vram_mb'
    ]
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
            
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rounded = {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()}
        writer.writerow(rounded)

def main():
    args = parse_args()
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    # Init distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    num_nodes = max(1, world_size // torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    
    hostname = socket.gethostname()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    
    # Check VRAM capacity
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_gb = mem_info.total / (1024**3)
    if args.target_gb > total_gb * 0.95:
        print(f"[WARNING] Rank {rank}: Requested {args.target_gb}GB is > 95% of total VRAM ({total_gb:.2f}GB). This may cause OOM.")

    # Pre-allocation with explicit error
    _filler = stress_gpu_memory(args.target_gb)
    if _filler is None and args.target_gb > 0:
        print(f"[CRITICAL] Rank {rank} failed to allocate {args.target_gb}GB VRAM. This GPU will be idle during the test.")
    
    if rank == 0:
        print(f"[MAIN] Initializing WandB (Mode: {os.environ.get('WANDB_MODE', 'online')})...")
        try:
            wandb.init(project=args.project, name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.config.update(vars(args))
        except Exception as e:
            print(f"[ERROR] WandB init failed: {e}. Continuing without WandB.")
        
        print(f"[MAIN] Starting benchmark with {world_size} GPUs across {num_nodes} nodes. Reporting every 60 seconds.")

    print(f"[RANK {rank}] Entering main loop...")
    start_time = time.time()
    last_report_time = start_time
    loop_in_minute = 0
    
    # Store history for averaging
    history_keys = ["tflops", "net_bw", "disk_w", "disk_r", "cpu_util", "ram_gb", "temp", "power", "util", "vram"]
    minute_history = {k: [] for k in history_keys}
    global_history = {k: [] for k in history_keys} 

    while (time.time() - start_time) < args.duration:
        loop_in_minute += 1
        
        # 1. GPU & Network Tests
        tflops = test_gpu_efficiency(args.gemm_size)
        net_bw = test_network_bw(args.net_size_mb)
        
        # 2. System Metrics
        cpu_util = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        ram_gb = ram_info.used / (1024**3)
        
        # 3. Disk I/O
        disk_w, disk_r = 0.0, 0.0
        # if local_rank == 0:
        #     disk_w, disk_r = test_disk_io()
            
        # 4. GPU Stats
        temp, pwr, util, vram = get_gpu_stats(handle)
        
        current_vals = [tflops, net_bw, disk_w, disk_r, cpu_util, ram_gb, temp, pwr, util, vram]
        for k, v in zip(history_keys, current_vals):
            minute_history[k].append(v)
            
        current_time = time.time()
        if (current_time - last_report_time) >= 60.0 or (current_time - start_time) >= args.duration:
            if len(minute_history["tflops"]) == 0:
                break
                
            local_minute_avg = [float(np.mean(minute_history[k])) for k in history_keys]
            metrics = torch.tensor(local_minute_avg, device='cuda')
            
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            if rank == 0:
                elapsed_mins = max(1, int((current_time - start_time) / 60))
                
                avg_tflops = metrics[0].item() / world_size
                avg_net = metrics[1].item() / world_size
                avg_dw = metrics[2].item() / num_nodes
                avg_dr = metrics[3].item() / num_nodes
                avg_cpu = metrics[4].item() / world_size
                avg_ram = metrics[5].item() / world_size
                avg_temp = metrics[6].item() / world_size
                avg_pwr = metrics[7].item() / world_size
                avg_util = metrics[8].item() / world_size
                avg_vram = metrics[9].item() / world_size
                
                avg_metrics = {
                    "minute": elapsed_mins,
                    "avg_tflops": avg_tflops,
                    "avg_net_bw_gbs": avg_net,
                    "avg_disk_w_mbs": avg_dw,
                    "avg_disk_r_mbs": avg_dr,
                    "avg_cpu_util": avg_cpu,
                    "avg_ram_used_gb": avg_ram,
                    "avg_temp_c": avg_temp,
                    "avg_power_w": avg_pwr,
                    "avg_gpu_util_pct": avg_util,
                    "avg_vram_mb": avg_vram
                }
                
                try:
                    wandb.log(avg_metrics)
                except:
                    pass
                save_to_csv(avg_metrics)
                
                print(f"[{elapsed_mins} min] Loops/min: {loop_in_minute} | TFLOPS: {avg_tflops:.2f} | Net: {avg_net:.2f} GB/s | Disk W: {avg_dw:.0f} MB/s | CPU: {avg_cpu:.1f}% | Temp: {avg_temp:.1f}C")
                
                for idx, k in enumerate(history_keys):
                    global_history[k].append(metrics[idx].item() / (world_size if idx not in [2,3] else num_nodes))

            minute_history = {k: [] for k in history_keys}
            loop_in_minute = 0
            last_report_time = current_time

    if rank == 0:
        print("\n" + "="*40)
        print("FINAL RESULTS (AVERAGE ACROSS ALL MINUTES)")
        print("="*40)
        
        final_metrics = {}
        out_keys = ["final_avg_tflops", "final_avg_net_bw_gbs", "final_avg_disk_w_mbs", "final_avg_disk_r_mbs", 
                    "final_avg_cpu_util", "final_avg_ram_used_gb", "final_avg_temp_c", "final_avg_power_w", 
                    "final_avg_gpu_util_pct", "final_avg_vram_mb"]
                    
        for idx, k in enumerate(history_keys):
            val = float(np.mean(global_history[k])) if len(global_history[k]) > 0 else 0.0
            final_metrics[out_keys[idx]] = val
            print(f"{out_keys[idx]:25}: {val:.2f}")
            try:
                wandb.run.summary[out_keys[idx]] = val
            except:
                pass
            
        print("="*40)
        try:
            wandb.finish()
        except:
            pass

    pynvml.nvmlShutdown()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
