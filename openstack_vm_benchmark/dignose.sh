#!/bin/bash
# diagnose.sh — run this on BOTH vm201 and vm202 before anything else
# Output tells us exactly why TFLOPS is low and P2P is wrong

echo "=============================="
echo " SYSTEM: $(hostname)"
echo " DATE:   $(date)"
echo "=============================="

echo ""
echo "── 1. GPU CLOCKS & POWER LIMITS ──────────────────────────"
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,power.limit,power.draw,pstate,persistence_mode \
  --format=csv,noheader
echo ""
echo "  (H200 should be: clocks.gr~1980MHz, power.limit=700W, pstate=P0)"

echo ""
echo "── 2. CLOCK THROTTLE REASONS ──────────────────────────────"
nvidia-smi --query-gpu=index,clocks_throttle_reasons.active \
  --format=csv,noheader
echo "  (should all be 'Not Active' or 'GPU Idle' when idle)"

echo ""
echo "── 3. APPLICATION CLOCK LOCKS ─────────────────────────────"
for i in $(seq 0 7); do
  echo -n "  GPU $i: "
  nvidia-smi -i $i --query-gpu=clocks.applications.gr,clocks.applications.mem \
    --format=csv,noheader 2>/dev/null || echo "N/A"
done

echo ""
echo "── 4. MIG MODE ─────────────────────────────────────────────"
nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader
echo "  (should all be 'Disabled')"

echo ""
echo "── 5. PERSISTENCE MODE ─────────────────────────────────────"
nvidia-smi --query-gpu=index,persistence_mode --format=csv,noheader
echo "  (should all be 'Enabled' for low-latency startup)"

echo ""
echo "── 6. ECC / RETIRED PAGES ──────────────────────────────────"
nvidia-smi --query-gpu=index,ecc.mode.current,retired_pages.sbe,retired_pages.dbe \
  --format=csv,noheader

echo ""
echo "── 7. QUICK SINGLE-GPU GEMM (no distributed) ──────────────"
python3 - << 'PYEOF'
import torch, time
torch.cuda.set_device(0)
N = 16384
a = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')
b = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')
c = torch.empty(N, N, dtype=torch.bfloat16, device='cuda')
# warmup
for _ in range(5): torch.matmul(a, b, out=c)
torch.cuda.synchronize()
iters = 20
t0 = time.perf_counter()
for _ in range(iters): torch.matmul(a, b, out=c)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
tflops = 2 * N**3 * iters / elapsed / 1e12
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
pwr = pynvml.nvmlDeviceGetPowerUsage(h)/1000
clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
print(f"  GPU 0 solo GEMM: {tflops:.1f} TFLOPS  Power={pwr:.0f}W  Clock={clk}MHz")
print(f"  Expected H200: ~3000+ TFLOPS, ~650-700W, ~1980MHz")
if tflops < 1000:
    print(f"  *** LOW: likely clock/power cap or wrong pstate ***")
PYEOF

echo ""
echo "── 8. ALL 8 GPUS PARALLEL GEMM (no distributed) ──────────"
python3 - << 'PYEOF'
import torch, time
from concurrent.futures import ThreadPoolExecutor

def run_gpu(dev):
    torch.cuda.set_device(dev)
    N = 8192
    a = torch.randn(N, N, dtype=torch.bfloat16, device=f'cuda:{dev}')
    b = torch.randn(N, N, dtype=torch.bfloat16, device=f'cuda:{dev}')
    c = torch.empty(N, N, dtype=torch.bfloat16, device=f'cuda:{dev}')
    for _ in range(3): torch.matmul(a, b, out=c)
    torch.cuda.synchronize(dev)
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters): torch.matmul(a, b, out=c)
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    tflops = 2 * N**3 * iters / elapsed / 1e12
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    pwr = pynvml.nvmlDeviceGetPowerUsage(h)/1000
    return dev, tflops, pwr

n = torch.cuda.device_count()
with ThreadPoolExecutor(max_workers=n) as ex:
    results = list(ex.map(run_gpu, range(n)))
for dev, tflops, pwr in results:
    flag = " *** LOW ***" if tflops < 800 else ""
    print(f"  GPU {dev}: {tflops:7.1f} TFLOPS  {pwr:4.0f}W{flag}")
PYEOF

echo ""
echo "── 9. IB DEVICES & STATE ───────────────────────────────────"
ibstat 2>/dev/null | grep -E "CA |State|Rate|Link" | head -40
echo ""
ibv_devinfo 2>/dev/null | grep -E "hca_id|port|state|active_mtu|link_layer" | head -40

echo ""
echo "── 10. IB RAW BANDWIDTH (run on vm202 first, then vm201) ───"
echo "  On vm202: ib_write_bw -a -F --ib-dev=mlx5_0"
echo "  On vm201: ib_write_bw -a -F --ib-dev=mlx5_0 <vm202_IB_IP>"
echo "  Expected NDR: ~46-50 GB/s at 4MB+ message size"

echo ""
echo "── 11. NCCL ENV CHECK ──────────────────────────────────────"
env | grep -E "NCCL|CUDA|TORCH" | sort

echo ""
echo "=============================="
echo " DONE. Share this output."
echo "=============================="
