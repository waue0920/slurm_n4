#!/usr/bin/env python3
"""
Benchmark Script: IO / GPU / InfiniBand
Target: 8x H200 (80GB HBM3), InfiniBand HDR/NDR
Output: result.json
"""

import os
import sys
import json
import time
import socket
import platform
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────
# Optional imports (graceful degradation)
# ──────────────────────────────────────────────
try:
    import torch
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not found – GPU / NCCL tests will be skipped")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ══════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════
CFG = {
    # IO
    "io_file_sizes_mb":   [256, 1024, 4096],   # sequential read/write per file size
    "io_block_size_kb":   4096,                  # 4 MB blocks
    "io_test_dir":        "/tmp/bench_io",

    # GPU
    "gpu_matmul_sizes":   [4096, 8192, 16384],  # square matrix dims
    "gpu_warmup_iters":   5,
    "gpu_bench_iters":    20,
    "gpu_memcpy_sizes_mb":[256, 1024, 4096],    # H2D / D2H / D2D

    # InfiniBand / NCCL (multi-GPU)
    "ib_allreduce_sizes_mb": [64, 256, 1024, 4096],
    "ib_bench_iters":      10,

    # P2P bandwidth (GPU ↔ GPU)
    "p2p_sizes_mb":       [256, 1024, 4096],
    "p2p_bench_iters":    10,
}

RESULTS = {
    "meta": {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "hostname":   socket.gethostname(),
        "platform":   platform.platform(),
        "python":     sys.version,
        "pytorch":    torch.__version__ if HAS_TORCH else None,
    },
    "io":        {},
    "gpu":       {},
    "p2p":       {},
    "infiniband": {},
    "errors":    [],
}


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def banner(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def log_error(section, msg):
    entry = {"section": section, "error": str(msg)}
    RESULTS["errors"].append(entry)
    print(f"  [ERROR] {section}: {msg}")

def gb(n_bytes):
    return n_bytes / (1024**3)

def mb(n_bytes):
    return n_bytes / (1024**2)


# ══════════════════════════════════════════════
#  1. DISK I/O BENCHMARK
# ══════════════════════════════════════════════
def bench_io():
    banner("DISK I/O BENCHMARK")
    io_dir = Path(CFG["io_test_dir"])
    io_dir.mkdir(parents=True, exist_ok=True)
    block = CFG["io_block_size_kb"] * 1024
    results = {}

    for size_mb in CFG["io_file_sizes_mb"]:
        n_bytes  = size_mb * 1024 * 1024
        n_blocks = max(1, n_bytes // block)
        fpath    = io_dir / f"bench_{size_mb}mb.bin"
        key      = f"{size_mb}MB"
        results[key] = {}

        # ── Sequential Write ──
        try:
            data = os.urandom(block)
            t0 = time.perf_counter()
            with open(fpath, "wb") as f:
                for _ in range(n_blocks):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
            elapsed = time.perf_counter() - t0
            bw_gbps = gb(n_blocks * block) / elapsed
            results[key]["write_GB_s"] = round(bw_gbps, 3)
            print(f"  Write {key}: {bw_gbps:.3f} GB/s  ({elapsed:.2f}s)")
        except Exception as e:
            log_error("io_write", e)

        # ── Sequential Read ──
        try:
            t0 = time.perf_counter()
            with open(fpath, "rb") as f:
                while f.read(block):
                    pass
            elapsed = time.perf_counter() - t0
            bw_gbps = gb(n_bytes) / elapsed
            results[key]["read_GB_s"] = round(bw_gbps, 3)
            print(f"  Read  {key}: {bw_gbps:.3f} GB/s  ({elapsed:.2f}s)")
        except Exception as e:
            log_error("io_read", e)

        # cleanup
        fpath.unlink(missing_ok=True)

    RESULTS["io"] = results


# ══════════════════════════════════════════════
#  2. GPU BENCHMARK  (per-device)
# ══════════════════════════════════════════════
def bench_gpu_single(dev_id):
    """MatMul TFLOPS + H2D/D2H/D2D bandwidth for one GPU."""
    dev = torch.device(f"cuda:{dev_id}")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev_id)
    res = {
        "name":        props.name,
        "vram_GB":     round(props.total_memory / 1024**3, 1),
        "sm_count":    props.multi_processor_count,
        "matmul_TF32": {},
        "matmul_BF16": {},
        "memcpy":      {},
    }

    # ── MatMul benchmark (TF32 + BF16) ──
    for dtype_name, dtype in [("TF32", torch.float32), ("BF16", torch.bfloat16)]:
        key = f"matmul_{dtype_name}"
        for N in CFG["gpu_matmul_sizes"]:
            try:
                a = torch.randn(N, N, dtype=dtype, device=dev)
                b = torch.randn(N, N, dtype=dtype, device=dev)
                # warmup
                for _ in range(CFG["gpu_warmup_iters"]):
                    c = torch.mm(a, b)
                torch.cuda.synchronize(dev)
                # bench
                t0 = time.perf_counter()
                for _ in range(CFG["gpu_bench_iters"]):
                    c = torch.mm(a, b)
                torch.cuda.synchronize(dev)
                elapsed = time.perf_counter() - t0
                flops   = 2 * N**3 * CFG["gpu_bench_iters"]   # FMA = 2 ops
                tflops  = flops / elapsed / 1e12
                res[key][f"N{N}"] = round(tflops, 2)
                print(f"  GPU{dev_id} {dtype_name} matmul N={N}: {tflops:.2f} TFLOPS")
                del a, b, c
            except Exception as e:
                log_error(f"gpu_{dev_id}_matmul_{dtype_name}_N{N}", e)

    # ── Memory bandwidth (H2D / D2H / D2D) ──
    for size_mb in CFG["gpu_memcpy_sizes_mb"]:
        n_bytes = size_mb * 1024 * 1024
        n_elem  = n_bytes // 4   # float32
        sub = {}
        try:
            cpu_t = torch.zeros(n_elem, dtype=torch.float32, pin_memory=True)
            gpu_t = torch.zeros(n_elem, dtype=torch.float32, device=dev)

            # H2D
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(CFG["gpu_bench_iters"]):
                gpu_t.copy_(cpu_t, non_blocking=False)
            torch.cuda.synchronize(dev)
            bw = gb(n_bytes * CFG["gpu_bench_iters"]) / (time.perf_counter() - t0)
            sub["H2D_GB_s"] = round(bw, 2)
            print(f"  GPU{dev_id} H2D {size_mb}MB: {bw:.2f} GB/s")

            # D2H
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(CFG["gpu_bench_iters"]):
                cpu_t.copy_(gpu_t, non_blocking=False)
            torch.cuda.synchronize(dev)
            bw = gb(n_bytes * CFG["gpu_bench_iters"]) / (time.perf_counter() - t0)
            sub["D2H_GB_s"] = round(bw, 2)
            print(f"  GPU{dev_id} D2H {size_mb}MB: {bw:.2f} GB/s")

            # D2D
            gpu_t2 = torch.zeros(n_elem, dtype=torch.float32, device=dev)
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(CFG["gpu_bench_iters"]):
                gpu_t2.copy_(gpu_t, non_blocking=False)
            torch.cuda.synchronize(dev)
            bw = gb(n_bytes * CFG["gpu_bench_iters"]) / (time.perf_counter() - t0)
            sub["D2D_GB_s"] = round(bw, 2)
            print(f"  GPU{dev_id} D2D {size_mb}MB: {bw:.2f} GB/s")

            res["memcpy"][f"{size_mb}MB"] = sub
            del cpu_t, gpu_t, gpu_t2
        except Exception as e:
            log_error(f"gpu_{dev_id}_memcpy_{size_mb}MB", e)

    torch.cuda.empty_cache()
    return res


def bench_gpu():
    banner("GPU BENCHMARK  (per-device)")
    if not HAS_TORCH:
        log_error("gpu", "PyTorch not available"); return
    if not torch.cuda.is_available():
        log_error("gpu", "CUDA not available"); return

    n_gpu = torch.cuda.device_count()
    print(f"  Detected {n_gpu} GPU(s)")
    RESULTS["gpu"]["device_count"] = n_gpu

    # Enable TF32 for H200 (Hopper)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for i in range(n_gpu):
        banner(f"  GPU {i}")
        try:
            RESULTS["gpu"][f"gpu{i}"] = bench_gpu_single(i)
        except Exception as e:
            log_error(f"gpu_{i}", e)


# ══════════════════════════════════════════════
#  3. PEER-TO-PEER (NVLink / PCIe) BANDWIDTH
# ══════════════════════════════════════════════
def bench_p2p():
    banner("PEER-TO-PEER GPU BANDWIDTH")
    if not HAS_TORCH or not torch.cuda.is_available():
        log_error("p2p", "CUDA not available"); return

    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        print("  Only 1 GPU detected – skipping P2P test"); return

    results = {}
    for src in range(n_gpu):
        for dst in range(n_gpu):
            if src == dst:
                continue
            key = f"gpu{src}→gpu{dst}"
            try:
                can_access = torch.cuda.can_device_access_peer(src, dst)
                if can_access:
                    torch.cuda.device(src)
                    torch.cuda.enable_peer_access(dst)
            except Exception:
                can_access = False

            for size_mb in CFG["p2p_sizes_mb"]:
                n_elem = (size_mb * 1024 * 1024) // 4
                try:
                    a = torch.zeros(n_elem, dtype=torch.float32,
                                    device=torch.device(f"cuda:{src}"))
                    b = torch.zeros(n_elem, dtype=torch.float32,
                                    device=torch.device(f"cuda:{dst}"))
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(CFG["p2p_bench_iters"]):
                        b.copy_(a, non_blocking=False)
                    torch.cuda.synchronize()
                    bw = gb(n_elem * 4 * CFG["p2p_bench_iters"]) / (time.perf_counter() - t0)
                    if key not in results:
                        results[key] = {"peer_access": can_access}
                    results[key][f"{size_mb}MB_GB_s"] = round(bw, 2)
                    print(f"  {key} {size_mb}MB: {bw:.2f} GB/s  (peer={can_access})")
                    del a, b
                except Exception as e:
                    log_error(f"p2p_{key}_{size_mb}MB", e)

    RESULTS["p2p"] = results


# ══════════════════════════════════════════════
#  4. INFINIBAND / NCCL ALL-REDUCE BENCHMARK
# ══════════════════════════════════════════════
def bench_infiniband():
    """
    Uses NCCL all-reduce across all GPUs as a proxy for IB bandwidth.
    Also queries `ibstat` / `ibv_devinfo` for HW info if available.
    """
    banner("INFINIBAND / NCCL ALL-REDUCE BENCHMARK")

    # ── Hardware info via ibstat ──
    ib_info = {}
    for cmd in ["ibstat", "ibv_devinfo"]:
        try:
            out = subprocess.check_output([cmd], stderr=subprocess.DEVNULL,
                                          timeout=10).decode(errors="replace")
            ib_info[cmd] = out[:4000]   # trim
            print(f"  {cmd}: captured ({len(out)} bytes)")
        except Exception:
            pass

    # Parse active port speeds from ibstat
    active_speeds = []
    for line in ib_info.get("ibstat", "").splitlines():
        if "Rate:" in line:
            active_speeds.append(line.strip())
    if active_speeds:
        ib_info["active_port_rates"] = active_speeds

    RESULTS["infiniband"]["hw_info"] = ib_info

    # ── NCCL all-reduce ──
    if not HAS_TORCH or not torch.cuda.is_available():
        log_error("ib_nccl", "PyTorch/CUDA not available"); return

    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        print("  Only 1 GPU – skipping NCCL all-reduce test"); return

    # Check if NCCL backend is available
    if not dist.is_nccl_available():
        log_error("ib_nccl", "NCCL backend not available"); return

    # We'll run the all-reduce in the main process using a process group
    # initialized over the loopback (single-node multi-GPU)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE",  str(n_gpu))
    os.environ.setdefault("RANK",        "0")

    # For single-node multi-GPU we can use torch.multiprocessing
    import torch.multiprocessing as mp
    results_queue = mp.Queue()
    mp.spawn(_nccl_worker, args=(n_gpu, results_queue), nprocs=n_gpu, join=True)

    # Collect results from rank 0
    if not results_queue.empty():
        RESULTS["infiniband"]["nccl_allreduce"] = results_queue.get()


def _nccl_worker(rank, world_size, results_queue):
    """Worker function spawned per GPU for NCCL all-reduce benchmark."""
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        results = {}
        for size_mb in CFG["ib_allreduce_sizes_mb"]:
            n_elem = (size_mb * 1024 * 1024) // 4
            tensor = torch.ones(n_elem, dtype=torch.float32,
                                device=torch.device(f"cuda:{rank}"))

            # Warmup
            for _ in range(3):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(CFG["ib_bench_iters"]):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - t0

            # Bus bandwidth formula: 2*(n-1)/n * size / time
            n   = world_size
            bw  = 2 * (n - 1) / n * gb(n_elem * 4 * CFG["ib_bench_iters"]) / elapsed
            alg_bw = gb(n_elem * 4 * CFG["ib_bench_iters"]) / elapsed

            if rank == 0:
                key = f"{size_mb}MB"
                results[key] = {
                    "bus_BW_GB_s": round(bw, 3),
                    "alg_BW_GB_s": round(alg_bw, 3),
                    "elapsed_s":   round(elapsed, 4),
                    "iters":       CFG["ib_bench_iters"],
                    "gpus":        world_size,
                }
                print(f"  NCCL AllReduce {key}: bus_BW={bw:.3f} GB/s  alg_BW={alg_bw:.3f} GB/s")
            del tensor

        if rank == 0:
            results_queue.put(results)

        dist.destroy_process_group()
    except Exception as e:
        if rank == 0:
            results_queue.put({"error": str(e)})
        print(f"  [ERROR] NCCL worker rank {rank}: {e}")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   8x H200 Benchmark: IO / GPU / InfiniBand  ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Host:  {RESULTS['meta']['hostname']}")
    print(f"  Time:  {RESULTS['meta']['timestamp']}")

    # Run benchmarks
    bench_io()
    bench_gpu()
    bench_p2p()
    bench_infiniband()

    # ── Write result.json ──
    out_path = Path("result.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print(f"\n{'═'*60}")
    print(f"  Results saved → {out_path.resolve()}")
    print(f"  Errors logged: {len(RESULTS['errors'])}")
    if RESULTS["errors"]:
        for e in RESULTS["errors"]:
            print(f"    • [{e['section']}] {e['error']}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    # Required for mp.spawn on some systems
    import torch.multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
    main()
