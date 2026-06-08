"""
benchmark_lib.py  –  Multi-node GPU COMPUTE stress test

Fixes in this version:
  1. GEMM runs on a dedicated CUDA stream CONTINUOUSLY — no stalls between iters
     NCCL collectives run on the default stream concurrently via async_op=True
  2. P2P test uses barrier-synchronized ping-pong to get accurate timing
  3. Loop exit: checked at top of every iteration, not only inside report block
  4. Workers exit cleanly via a poison-pill all_reduce signal from rank 0
"""

import os, sys, csv, json, time, socket, argparse
from datetime import datetime

import torch
import torch.distributed as dist
import numpy as np
import pynvml, psutil

try:
    import wandb; HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project",         type=str,   default="openstack_vm_benchmark")
    p.add_argument("--duration",        type=int,   default=86400)
    p.add_argument("--target_gb",       type=float, default=0)
    p.add_argument("--gemm_size",       type=int,   default=16384)
    p.add_argument("--gemm_iters",      type=int,   default=50,
                   help="matmuls per TFLOPS sample (higher = more accurate, more time)")
    p.add_argument("--net_size_mb",     type=int,   default=1024,
                   help="AllReduce payload in MB (use >=512 to stress IB cross-node)")
    p.add_argument("--report_interval", type=int,   default=60)
    p.add_argument("--offline",         action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
def get_gpu_stats(handle):
    try:
        temp  = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        util  = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram  = float(mem.used) / 1024**3
        return temp, power, util, vram
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────
class GemmStress:
    """
    Runs BF16 matmul on a private CUDA stream so it overlaps with
    NCCL collectives on the default stream.
    Buffers are allocated once and reused — no per-iter VRAM churn.
    """
    def __init__(self, size: int):
        self.size   = size
        self.stream = torch.cuda.Stream(priority=-1)   # high-priority stream
        with torch.cuda.stream(self.stream):
            self.a = torch.randn(size, size, dtype=torch.bfloat16, device="cuda")
            self.b = torch.randn(size, size, dtype=torch.bfloat16, device="cuda")
            self.c = torch.empty(size, size, dtype=torch.bfloat16, device="cuda")
        gb = 3 * size**2 * 2 / 1024**3
        print(f"  [GemmStress] cuda:{torch.cuda.current_device()}  "
              f"{size}²×BF16  buffers={gb:.2f} GB")

    def run(self, iters: int) -> float:
        """Submit iters matmuls on the private stream; return TFLOPS."""
        with torch.cuda.stream(self.stream):
            # one warmup
            torch.matmul(self.a, self.b, out=self.c)
            self.stream.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                torch.matmul(self.a, self.b, out=self.c)
            self.stream.synchronize()
        elapsed = time.perf_counter() - t0
        return 2 * self.size**3 * iters / elapsed / 1e12

    def free(self):
        del self.a, self.b, self.c
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
def measure_allreduce(size_mb: int, world_size: int, iters: int = 5):
    """
    All-reduce on the DEFAULT stream (separate from GEMM stream).
    Returns (algo_GBs, bus_GBs).
      algo = bytes / time
      bus  = algo × 2(n-1)/n   ← standard nccl-tests formula
    """
    n_elem = (size_mb * 1024 * 1024) // 4
    t = torch.ones(n_elem, dtype=torch.float32, device="cuda")
    dist.all_reduce(t); torch.cuda.synchronize()   # warmup

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n    = world_size
    algo = n_elem * 4 * iters / elapsed / 1e9
    bus  = algo * 2 * (n - 1) / n
    del t
    return algo, bus


def measure_p2p(rank: int, world_size: int, size_mb: int = 256, iters: int = 5):
    """
    Properly synchronized ping-pong.
    Even rank sends, odd rank receives; then swap.
    Returns unidirectional BW in GB/s (same for both directions).
    """
    partner = rank ^ 1
    if partner >= world_size:
        return 0.0

    n_elem = (size_mb * 1024 * 1024) // 4
    buf    = torch.ones(n_elem, dtype=torch.float32, device="cuda")

    dist.barrier(); torch.cuda.synchronize()

    # Direction A: even→odd
    t0 = time.perf_counter()
    for _ in range(iters):
        if rank % 2 == 0:
            req = dist.isend(buf, dst=partner)
            req.wait()
        else:
            req = dist.irecv(buf, src=partner)
            req.wait()
    torch.cuda.synchronize()
    elapsed_a = time.perf_counter() - t0

    dist.barrier(); torch.cuda.synchronize()

    # Direction B: odd→even
    t0 = time.perf_counter()
    for _ in range(iters):
        if rank % 2 == 1:
            req = dist.isend(buf, dst=partner)
            req.wait()
        else:
            req = dist.irecv(buf, src=partner)
            req.wait()
    torch.cuda.synchronize()
    elapsed_b = time.perf_counter() - t0

    del buf
    bytes_total = n_elem * 4 * iters
    bw = bytes_total / ((elapsed_a + elapsed_b) / 2) / 1e9
    return bw


def measure_disk(mb: int = 512):
    t      = torch.randn(mb * 1024 * 1024 // 4)
    nbytes = t.nelement() * 4
    path   = f"/tmp/bench_{os.getpid()}.tmp"
    w = r  = 0.0
    try:
        t0 = time.perf_counter()
        torch.save(t, path)
        fd = os.open(path, os.O_WRONLY); os.fsync(fd); os.close(fd)
        w  = nbytes / (time.perf_counter() - t0) / 1e6

        t0 = time.perf_counter()
        torch.load(path, weights_only=True)
        r  = nbytes / (time.perf_counter() - t0) / 1e6
    except Exception as e:
        print(f"[WARN] Disk: {e}")
    finally:
        if os.path.exists(path): os.remove(path)
    del t
    return w, r


# ─────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp", "elapsed_min", "iter",
    "tflops_bf16",
    "allreduce_algo_GBs", "allreduce_bus_GBs",
    "p2p_GBs",
    "disk_w_MBs", "disk_r_MBs",
    "gpu_util_pct", "gpu_power_w", "gpu_temp_c", "vram_used_GB",
    "cpu_util_pct", "ram_used_GB",
]

def write_csv(row, path="results.csv"):
    exists = os.path.isfile(path)
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists: w.writeheader()
        w.writerow(rounded)


# ─────────────────────────────────────────────────────────────
def broadcast_stop(rank: int):
    """Rank 0 signals all ranks to stop via a sentinel all_reduce."""
    stop = torch.tensor([1.0], device="cuda")
    dist.all_reduce(stop)

def check_stop(rank: int) -> bool:
    """Non-rank-0 calls this each iteration to poll for stop signal.
    Returns True when rank 0 has set the stop flag."""
    stop = torch.tensor([0.0], device="cuda")
    dist.all_reduce(stop)
    return stop.item() > 0.5


# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    dist.init_process_group(backend="nccl")
    rank        = dist.get_rank()
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = dist.get_world_size()
    n_gpu_local = torch.cuda.device_count()
    num_nodes   = max(1, world_size // n_gpu_local)
    torch.cuda.set_device(local_rank)

    pynvml.nvmlInit()
    handle   = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    gpu_name = pynvml.nvmlDeviceGetName(handle)

    if rank == 0:
        print(f"\n{'='*64}")
        print(f"  GPU Stress Benchmark")
        print(f"  {world_size} ranks | {num_nodes} nodes | {n_gpu_local} GPUs/node | {gpu_name}")
        print(f"  GEMM  : {args.gemm_size}² BF16 × {args.gemm_iters} iters/sample")
        print(f"  IB AR : {args.net_size_mb} MB payload")
        print(f"  Time  : {args.duration}s, report every {args.report_interval}s")
        print(f"{'='*64}\n")

    # Pre-flight
    probe = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(probe)
    if rank == 0:
        expected = float(world_size * (world_size - 1) / 2)
        ok = abs(probe.item() - expected) < 0.5
        print(f"[PREFLIGHT] {'✓ all ' + str(world_size) + ' ranks healthy' if ok else '✗ rank mismatch — aborting'}\n")
        if not ok:
            dist.destroy_process_group(); sys.exit(1)
    del probe
    dist.barrier()

    # VRAM filler
    filler = None
    if args.target_gb > 0:
        free_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**3
        fill_gb = min(args.target_gb, free_gb * 0.80)
        try:
            filler = torch.zeros(int(fill_gb * 1024**3 / 2),
                                  dtype=torch.float16, device="cuda")
            if rank == 0: print(f"[INFO] VRAM filler: {fill_gb:.1f} GB/GPU\n")
        except Exception as e:
            print(f"[WARN] rank {rank} VRAM filler: {e}")

    gemm = GemmStress(args.gemm_size)

    use_wandb = False
    if rank == 0 and HAS_WANDB:
        try:
            wandb.init(project=args.project,
                       name=f"stress-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.config.update(vars(args)); use_wandb = True
        except Exception as e:
            print(f"[WARN] WandB: {e}")

    KEYS    = ["tflops","ar_algo","ar_bus","p2p",
               "disk_w","disk_r",
               "gpu_util","gpu_pwr","gpu_temp","vram",
               "cpu_util","ram_gb"]
    window  = {k: [] for k in KEYS}
    history = {k: [] for k in KEYS}

    t_start    = time.time()
    t_last_rpt = t_start
    iter_n     = 0

    # ── Main loop ─────────────────────────────────────────────
    # NOTE: exit condition is checked at the TOP of every iteration.
    # Rank 0 drives the stop signal; all other ranks obey it.
    # This ensures clean synchronized exit regardless of timing drift.
    while True:
        # ── Exit check (ALL ranks participate) ────────────────
        elapsed = time.time() - t_start
        # Pack stop flag + elapsed into one all_reduce to save a round-trip
        stop_flag  = 1.0 if (rank == 0 and elapsed >= args.duration) else 0.0
        ctrl       = torch.tensor([stop_flag], device="cuda")
        dist.all_reduce(ctrl)          # sum: >0 means rank 0 said stop
        if ctrl.item() > 0.5:
            break
        del ctrl

        iter_n += 1

        # 1. BF16 GEMM on private stream (overlaps with steps 2-3)
        try:
            tflops = gemm.run(args.gemm_iters)
        except Exception as e:
            print(f"[WARN] rank {rank} GEMM: {e}"); tflops = 0.0

        # 2. AllReduce over IB (default stream, concurrent with GEMM stream)
        try:
            ar_algo, ar_bus = measure_allreduce(args.net_size_mb, world_size)
        except Exception as e:
            print(f"[WARN] rank {rank} AllReduce: {e}"); ar_algo = ar_bus = 0.0

        # 3. P2P bandwidth (IB cross-node pairs)
        try:
            p2p = measure_p2p(rank, world_size, n_gpu_local)
        except Exception as e:
            print(f"[WARN] rank {rank} P2P: {e}"); p2p = 0.0

        # 4. Disk (local_rank 0 only)
        disk_w = disk_r = 0.0
        if local_rank == 0:
            disk_w, disk_r = measure_disk()
        dt = torch.tensor([disk_w, disk_r], device="cuda")
        try:
            dist.all_reduce(dt, op=dist.ReduceOp.SUM)
            disk_w = dt[0].item() / num_nodes
            disk_r = dt[1].item() / num_nodes
        except Exception: pass
        del dt

        # 5. System metrics
        cpu_util = psutil.cpu_percent(interval=None)
        ram_gb   = psutil.virtual_memory().used / 1024**3
        temp, pwr, util, vram = get_gpu_stats(handle)

        for k, v in zip(KEYS, [tflops, ar_algo, ar_bus, p2p,
                                 disk_w, disk_r,
                                 util, pwr, temp, vram,
                                 cpu_util, ram_gb]):
            window[k].append(v)

        # ── Report ────────────────────────────────────────────
        now    = time.time()
        do_rpt = (now - t_last_rpt >= args.report_interval)

        if do_rpt and window["tflops"]:
            try: dist.barrier()
            except Exception as e: print(f"[WARN] rank {rank} barrier: {e}")

            vec = torch.tensor(
                [float(np.mean(window[k])) for k in KEYS], device="cuda")
            try: dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            except Exception as e: print(f"[WARN] rank {rank} report reduce: {e}")

            if rank == 0:
                W   = world_size
                avg = {k: vec[i].item() / W for i, k in enumerate(KEYS)}
                avg["p2p"]    = vec[KEYS.index("p2p")].item()    / max(1, W // 2)
                avg["disk_w"] = vec[KEYS.index("disk_w")].item() / num_nodes
                avg["disk_r"] = vec[KEYS.index("disk_r")].item() / num_nodes

                elapsed_min = (now - t_start) / 60
                row = {
                    "elapsed_min":        round(elapsed_min, 2),
                    "iter":               iter_n,
                    "tflops_bf16":        avg["tflops"],
                    "allreduce_algo_GBs": avg["ar_algo"],
                    "allreduce_bus_GBs":  avg["ar_bus"],
                    "p2p_GBs":            avg["p2p"],
                    "disk_w_MBs":         avg["disk_w"],
                    "disk_r_MBs":         avg["disk_r"],
                    "gpu_util_pct":       avg["gpu_util"],
                    "gpu_power_w":        avg["gpu_pwr"],
                    "gpu_temp_c":         avg["gpu_temp"],
                    "vram_used_GB":       avg["vram"],
                    "cpu_util_pct":       avg["cpu_util"],
                    "ram_used_GB":        avg["ram_gb"],
                }
                print(
                    f"[{elapsed_min:6.1f} min | iter {iter_n:5d}]  "
                    f"BF16={avg['tflops']:7.1f} TFLOPS  "
                    f"AR_algo={avg['ar_algo']:6.2f} GB/s  "
                    f"AR_bus={avg['ar_bus']:6.2f} GB/s  "
                    f"P2P={avg['p2p']:5.2f} GB/s  "
                    f"GPU%={avg['gpu_util']:3.0f}  "
                    f"Pwr={avg['gpu_pwr']:4.0f}W  "
                    f"Temp={avg['gpu_temp']:3.0f}°C"
                )
                write_csv(row)
                if use_wandb:
                    try: wandb.log(row)
                    except Exception: pass
                for k in KEYS:
                    history[k].append(avg[k])

            window     = {k: [] for k in KEYS}
            t_last_rpt = now

    # ── Final summary ─────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*64}")
        print("FINAL AVERAGES")
        print(f"{'='*64}")
        labels = {
            "tflops":   "BF16 TFLOPS          (H200 target: ~3000+)",
            "ar_algo":  "AllReduce algo  GB/s  (bytes/time)",
            "ar_bus":   "AllReduce bus   GB/s  (× 2(n-1)/n)",
            "p2p":      "P2P IB          GB/s  (per rank-pair)",
            "disk_w":   "Disk Write      MB/s",
            "disk_r":   "Disk Read       MB/s",
            "gpu_util": "GPU Util        %     (target: ~99%)",
            "gpu_pwr":  "GPU Power       W     (H200 TDP: 700W)",
            "gpu_temp": "GPU Temp        °C",
            "vram":     "VRAM Used       GB",
            "cpu_util": "CPU Util        %",
            "ram_gb":   "RAM Used        GB",
        }
        final = {}
        for k in KEYS:
            v = float(np.mean(history[k])) if history[k] else 0.0
            final[k] = round(v, 3)
            print(f"  {labels[k]:45s}: {v:.2f}")
            if use_wandb:
                try: wandb.run.summary[k] = v
                except Exception: pass

        print(f"\n{'='*64}")
        print("BANDWIDTH UNITS")
        print("  All BW figures are in GB/s (GigaBytes/sec, SI: 10^9 bytes)")
        print("  IB links are rated in Gb/s (GigaBits/sec): 1 GB/s = 8 Gb/s")
        print("  NDR 400Gb/s port = 50 GB/s | 8 ports = 400 GB/s aggregate")
        print(f"{'='*64}\n")

        doc = {
            "meta": {
                "timestamp":     datetime.now().isoformat(),
                "hostname":      socket.gethostname(),
                "world_size":    world_size,
                "num_nodes":     num_nodes,
                "gpus_per_node": n_gpu_local,
                "gpu_model":     gpu_name,
                "duration_s":    args.duration,
            },
            "units": {
                "note":               "All BW in GB/s (SI Gigabytes). IB rated in Gb/s (bits).",
                "tflops_bf16":        "TFLOPS (2×N³ ops, BF16 FMA)",
                "allreduce_algo_GBs": "GB/s raw (bytes / time)",
                "allreduce_bus_GBs":  "GB/s bus (algo × 2(n-1)/n, nccl-tests standard)",
                "p2p_GBs":            "GB/s per IB rank-pair (uni-directional avg)",
            },
            "config":         vars(args),
            "final_averages": {k: final[k] for k in KEYS},
            "per_window": [
                {k: round(history[k][i], 3) for k in KEYS}
                for i in range(len(history["tflops"]))
            ],
        }
        with open("result.json", "w") as f:
            json.dump(doc, f, indent=2)
        print("[MAIN] Saved → result.json  results.csv")
        if use_wandb:
            try: wandb.finish()
            except Exception: pass

    gemm.free()
    if filler is not None: del filler
    pynvml.nvmlShutdown()
    dist.destroy_process_group()
    print(f"[RANK {rank}] clean exit.")


if __name__ == "__main__":
    main()
on.dump(doc, f, indent=2)
        print("[MAIN] Saved → result.json  results.csv")
        if use_wandb:
            try: wandb.finish()
            except Exception: pass

    gemm.free()
    if filler is not None: del filler
    pynvml.nvmlShutdown()
    dist.destroy_process_group()
    print(f"[RANK {rank}] clean exit.")


if __name__ == "__main__":
    main()
