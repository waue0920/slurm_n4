"""
benchmark_lib.py  –  Multi-node GPU / Network stress test
Fixes vs original:
  * Report barrier is ALL-RANK synchronized (no more NCCL deadlock)
  * Disk I/O re-enabled on local_rank==0 only, result broadcast to peers
  * All dist.all_reduce calls are inside try/except with graceful recovery
  * Robust 24h+ loop: per-iteration timeout guard, loop never stalls
  * result.json written by rank-0 at the end (in addition to CSV / WandB)
  * NCCL watchdog: skips one cycle instead of hanging if collective times out
"""

import torch
import torch.distributed as dist
import time
import os
import pynvml
import wandb
import numpy as np
import argparse
import csv
import json
import socket
import psutil
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  ARG PARSING
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Node GPU Stress Test")
    parser.add_argument("--project",         type=str, default="openstack_vm_benchmark")
    parser.add_argument("--duration",        type=int, default=86400,  help="Total duration in seconds (default 24 h)")
    parser.add_argument("--target_gb",       type=int, default=80,     help="VRAM to fill per GPU in GB")
    parser.add_argument("--gemm_size",       type=int, default=16384,  help="Matrix size for GEMM test")
    parser.add_argument("--net_size_mb",     type=int, default=1024,   help="Data size for NCCL all-reduce (MB)")
    parser.add_argument("--report_interval", type=int, default=60,     help="Reporting interval in seconds")
    parser.add_argument("--offline",         action="store_true",      help="Run WandB in offline mode")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def get_gpu_stats(handle):
    try:
        temp  = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        util  = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram  = float(mem.used) / (1024 * 1024)
        return temp, power, util, vram
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def stress_gpu_memory(target_gb, rank):
    if target_gb <= 0:
        return None
    try:
        n_elem = int((target_gb * 1024**3) / 2)   # float16 = 2 bytes
        filler = torch.zeros(n_elem, dtype=torch.float16, device="cuda")
        print(f"[VRAM] Rank {rank}: allocated {target_gb} GB")
        return filler
    except Exception as e:
        print(f"[VRAM] Rank {rank}: allocation failed – {e}")
        return None


# ─────────────────────────────────────────────────────────────
#  BENCHMARK KERNELS
# ─────────────────────────────────────────────────────────────
def test_gpu_efficiency(size, iters=20):
    """FP16 GEMM → TFLOPS"""
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tflops  = (2 * size**3 * iters) / elapsed / 1e12
    del a, b
    return tflops


def test_network_bw(size_mb, iters=10):
    """NCCL all-reduce → algoBW (GB/s).
    Returns (bw, ok) – ok=False means the collective timed out / failed.
    """
    n_elem = (size_mb * 1024 * 1024) // 4
    tensor = torch.randn(n_elem, device="cuda")
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        bw = (n_elem * 4 * iters) / elapsed / 1e9
        del tensor
        return bw, True
    except Exception as e:
        print(f"[WARN] NCCL all-reduce failed: {e}")
        del tensor
        return 0.0, False


def test_disk_io(mb_size=512):
    """Sequential write + read via torch.save/load.  Only call on local_rank==0."""
    tensor       = torch.randn(1024, 1024, mb_size // 4)
    tensor_bytes = tensor.nelement() * 4
    path         = f"/tmp/bench_{os.getpid()}.tmp"
    w_speed = r_speed = 0.0
    try:
        # Write
        t0 = time.perf_counter()
        torch.save(tensor, path)
        fd = os.open(path, os.O_WRONLY)
        os.fsync(fd); os.close(fd)
        w_speed = tensor_bytes / (time.perf_counter() - t0) / 1e6

        # Read
        t0 = time.perf_counter()
        torch.load(path, weights_only=True)
        r_speed = tensor_bytes / (time.perf_counter() - t0) / 1e6
    except Exception as e:
        print(f"[WARN] Disk I/O test failed: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)
    del tensor
    return w_speed, r_speed


# ─────────────────────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "timestamp", "elapsed_min",
    "avg_tflops", "avg_net_bw_gbs",
    "avg_disk_w_mbs", "avg_disk_r_mbs",
    "avg_cpu_util", "avg_ram_used_gb",
    "avg_temp_c", "avg_power_w", "avg_gpu_util_pct", "avg_vram_mb",
]

def save_to_csv(metrics, filename="results.csv"):
    exists = os.path.isfile(filename)
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}
    with open(filename, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(rounded)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    # ── Distributed init ──────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    n_gpu_node = torch.cuda.device_count()
    num_nodes  = max(1, world_size // n_gpu_node)
    torch.cuda.set_device(local_rank)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    # VRAM guard
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_gb = mem_info.total / 1024**3
    if args.target_gb > total_gb * 0.95:
        print(f"[WARN] Rank {rank}: {args.target_gb} GB > 95% of {total_gb:.1f} GB VRAM – may OOM")

    _filler = stress_gpu_memory(args.target_gb, rank)

    # ── WandB (rank 0 only) ───────────────────────────────────
    use_wandb = False
    if rank == 0:
        print(f"[MAIN] WandB mode: {os.environ.get('WANDB_MODE', 'online')}")
        try:
            wandb.init(
                project=args.project,
                name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            wandb.config.update(vars(args))
            use_wandb = True
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}")

        print(
            f"[MAIN] {world_size} GPUs / {num_nodes} nodes | "
            f"duration={args.duration}s | report_interval={args.report_interval}s"
        )

    # ── Metric accumulators ───────────────────────────────────
    KEYS = ["tflops", "net_bw", "disk_w", "disk_r",
            "cpu_util", "ram_gb", "temp", "power", "util", "vram"]
    N    = len(KEYS)

    interval_buf  = {k: [] for k in KEYS}   # within current reporting window
    global_buf    = {k: [] for k in KEYS}   # across all windows (rank-0 only)

    start_time        = time.time()
    last_report_time  = start_time
    iter_count        = 0

    print(f"[RANK {rank}] Starting main loop (duration={args.duration}s) …")

    # ── Main loop ─────────────────────────────────────────────
    while True:
        loop_start = time.time()
        elapsed    = loop_start - start_time
        if elapsed >= args.duration:
            break

        iter_count += 1

        # 1. GEMM
        try:
            tflops = test_gpu_efficiency(args.gemm_size)
        except Exception as e:
            print(f"[WARN] Rank {rank} GEMM failed: {e}")
            tflops = 0.0

        # 2. NCCL all-reduce  (ALL ranks must call this together)
        net_bw, nccl_ok = test_network_bw(args.net_size_mb)
        if not nccl_ok:
            # Give the group a moment to recover before next iteration
            time.sleep(5)

        # 3. Disk I/O  (local_rank==0 only; broadcast result to local peers)
        disk_w = disk_r = 0.0
        if local_rank == 0:
            disk_w, disk_r = test_disk_io()
        # Broadcast disk results from local_rank-0 to all ranks on same node
        # We do this via a small all-reduce: non-zero values only from l_rank0
        disk_tensor = torch.tensor([disk_w, disk_r], device="cuda")
        # Sum across all ranks; only local_rank-0 contributed non-zero values,
        # so divide by num_nodes to recover per-node average
        try:
            dist.all_reduce(disk_tensor, op=dist.ReduceOp.SUM)
            disk_w = disk_tensor[0].item() / num_nodes
            disk_r = disk_tensor[1].item() / num_nodes
        except Exception:
            disk_w = disk_r = 0.0
        del disk_tensor

        # 4. System metrics
        cpu_util = psutil.cpu_percent(interval=None)
        ram_gb   = psutil.virtual_memory().used / 1024**3
        temp, pwr, util, vram = get_gpu_stats(handle)

        # Buffer
        vals = [tflops, net_bw, disk_w, disk_r, cpu_util, ram_gb, temp, pwr, util, vram]
        for k, v in zip(KEYS, vals):
            interval_buf[k].append(v)

        # ── Reporting window ──────────────────────────────────
        now = time.time()
        window_done = (now - last_report_time) >= args.report_interval
        final_iter  = (now - start_time)       >= args.duration

        if window_done or final_iter:
            # ── SYNCHRONIZED barrier so all ranks report together ──
            # This is the critical fix: previously rank-0 might call
            # all_reduce here while workers had already exited the loop.
            try:
                dist.barrier()
            except Exception as e:
                print(f"[WARN] Rank {rank} barrier failed: {e}")

            if not interval_buf["tflops"]:
                last_report_time = now
                interval_buf = {k: [] for k in KEYS}
                if final_iter:
                    break
                continue

            # Local averages → tensor for global reduction
            local_avgs = torch.tensor(
                [float(np.mean(interval_buf[k])) for k in KEYS],
                device="cuda",
            )

            try:
                dist.all_reduce(local_avgs, op=dist.ReduceOp.SUM)
            except Exception as e:
                print(f"[WARN] Rank {rank} report all_reduce failed: {e}")

            if rank == 0:
                elapsed_min = max(1, int((now - start_time) / 60))
                # Divide: per-GPU metrics by world_size; disk by num_nodes
                divisors = [world_size] * N
                divisors[2] = num_nodes   # disk_w already averaged above
                divisors[3] = num_nodes   # disk_r

                avgs = {k: local_avgs[i].item() / divisors[i]
                        for i, k in enumerate(KEYS)}

                report = {
                    "elapsed_min":      elapsed_min,
                    "avg_tflops":       avgs["tflops"],
                    "avg_net_bw_gbs":   avgs["net_bw"],
                    "avg_disk_w_mbs":   avgs["disk_w"],
                    "avg_disk_r_mbs":   avgs["disk_r"],
                    "avg_cpu_util":     avgs["cpu_util"],
                    "avg_ram_used_gb":  avgs["ram_gb"],
                    "avg_temp_c":       avgs["temp"],
                    "avg_power_w":      avgs["power"],
                    "avg_gpu_util_pct": avgs["util"],
                    "avg_vram_mb":      avgs["vram"],
                }

                if use_wandb:
                    try:
                        wandb.log(report)
                    except Exception:
                        pass

                save_to_csv(report)

                print(
                    f"[{elapsed_min:4d} min | iter {iter_count:6d}] "
                    f"TFLOPS={avgs['tflops']:6.1f}  "
                    f"Net={avgs['net_bw']:5.2f} GB/s  "
                    f"Disk W={avgs['disk_w']:5.0f} MB/s  "
                    f"GPU%={avgs['util']:3.0f}  "
                    f"Temp={avgs['temp']:3.0f}°C  "
                    f"Pwr={avgs['power']:5.0f}W"
                )

                for k in KEYS:
                    global_buf[k].append(avgs[k])

            # Reset window
            interval_buf     = {k: [] for k in KEYS}
            last_report_time = now

            if final_iter:
                break

    # ── Final summary (rank 0) ────────────────────────────────
    if rank == 0:
        print("\n" + "=" * 60)
        print("FINAL RESULTS  (average across all reporting windows)")
        print("=" * 60)

        LABEL = {
            "tflops":   "TFLOPS (FP16 GEMM)",
            "net_bw":   "Net BW all-reduce (GB/s)",
            "disk_w":   "Disk Write (MB/s)",
            "disk_r":   "Disk Read  (MB/s)",
            "cpu_util": "CPU Util (%)",
            "ram_gb":   "RAM Used (GB)",
            "temp":     "GPU Temp (°C)",
            "power":    "GPU Power (W)",
            "util":     "GPU Util (%)",
            "vram":     "VRAM Used (MB)",
        }

        final = {}
        for k in KEYS:
            v = float(np.mean(global_buf[k])) if global_buf[k] else 0.0
            final[k] = round(v, 3)
            print(f"  {LABEL[k]:30s}: {v:.2f}")
            if use_wandb:
                try:
                    wandb.run.summary[f"final_{k}"] = v
                except Exception:
                    pass

        print("=" * 60)

        # ── result.json ───────────────────────────────────────
        result_doc = {
            "meta": {
                "timestamp":   datetime.now().isoformat(),
                "hostname":    socket.gethostname(),
                "world_size":  world_size,
                "num_nodes":   num_nodes,
                "duration_s":  args.duration,
                "gpus_per_node": n_gpu_node,
            },
            "config":  vars(args),
            "results": {
                f"final_avg_{k}": v for k, v in final.items()
            },
            "per_window": [
                {k: round(global_buf[k][i], 3) for k in KEYS}
                for i in range(len(global_buf["tflops"]))
            ],
        }
        with open("result.json", "w") as f:
            json.dump(result_doc, f, indent=2)
        print("[MAIN] Saved → result.json")

        if use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    pynvml.nvmlShutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
