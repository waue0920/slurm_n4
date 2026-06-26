import torch

print("CUDA 可用:", torch.cuda.is_available())
print("GPU 名稱:", torch.cuda.get_device_name(0))
print("PyTorch CUDA 版本:", torch.version.cuda)
print("GPU 數量:", torch.cuda.device_count())

x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print("矩陣乘法結果 shape:", z.shape)
print("✅ GPU 運算成功！")
