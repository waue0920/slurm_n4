import torch
import nemo
import socket
import subprocess
import nemo

def print_versions():
    print("PyTorch version:", torch.__version__)
    print("NeMo version:", nemo.__version__)

def print_system_info():
    hostname = socket.gethostname()
    print(f"***[{hostname}]***")
    try:
        gpus = torch.cuda.device_count()
        print(f"GPUs:{gpus}")
    except Exception as e:
        print("Error getting GPU information:", e)

def print_cuda_info():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print("CUDA is available:", cuda_available)
    # Print CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("CUDA version:", result.stdout.splitlines()[-1])
    except FileNotFoundError:
        print("CUDA not installed or not found.")



if __name__ == "__main__":
    print_system_info()
    print_cuda_info()
    print_versions()

