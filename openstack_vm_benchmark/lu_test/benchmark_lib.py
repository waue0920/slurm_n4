import os
import time
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="NCCL Synthetic Bandwidth Benchmark")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--target_gb", type=float, default=40.0, help="Size of tensor in GB per GPU")
    parser.add_argument("--offline", action="store_true", help="Ignore wandb (dummy flag for compatibility)")
    args = parser.parse_args()

    # 1. Initialize Distributed Process Group
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Bind the process to the specific GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print("==========================================================")
        print(f"🚀 Starting PyTorch NCCL Benchmark | World Size: {world_size}")
        print(f"📦 Payload per GPU: {args.target_gb} GB | ⏱️ Target Duration: {args.duration} sec")
        print("==========================================================")

    # 2. Allocate VRAM (1 GB = 1024^3 bytes, float32 = 4 bytes per element)
    num_elements = int((args.target_gb * (1024**3)) / 4)
    
    if global_rank == 0:
        print(f"[Rank 0] Allocating {args.target_gb} GB of VRAM...")
        
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    dist.barrier()

    # 3. Warmup Phase (Crucial for establishing IB connections)
    if global_rank == 0:
        print("[Rank 0] Warming up NCCL and PCIe/IB pathways...")
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    if global_rank == 0:
        print("[Rank 0] 🔥 Starting benchmark loop...\n")

    # 4. Execution and Timing
    start_time = time.time()
    iterations = 0

    # CUDA events ensure we only measure true GPU execution time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    # We sync inside the loop to prevent the CPU from queuing thousands of async 
    # operations and drifting the time.time() measurement.
    while (time.time() - start_time) < args.duration:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        iterations += 1

    end_event.record()
    torch.cuda.synchronize()
    dist.barrier()

    # 5. Math and Reporting
    if global_rank == 0:
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_sec = elapsed_time_ms / 1000.0
        
        total_data_gb = args.target_gb * iterations
        algo_bw = total_data_gb / elapsed_time_sec
        
        correction_factor = 2 * (world_size - 1) / world_size
        bus_bw = algo_bw * correction_factor

        print("==========================================================")
        print("✅ Benchmark Complete")
        print("==========================================================")
        print(f"Total Iterations : {iterations}")
        print(f"Elapsed Time     : {elapsed_time_sec:.2f} seconds")
        print(f"Avg Time / Iter  : {(elapsed_time_sec / iterations) * 1000:.2f} ms")
        print("----------------------------------------------------------")
        print(f"Algorithmic BW   : {algo_bw:.2f} GB/s")
        print(f"Hardware Bus BW  : {bus_bw:.2f} GB/s")
        print("==========================================================")

    # Clean up
    del tensor
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
