# OpenStack 雲端環境：NVIDIA GPU (H200) 完整安裝與部署 SOP

* 此 SOP 專為 **OpenStack 虛擬化環境** 設計的驅動程式安裝設定
  - （實體機插滿 8 張 H200 物理卡並配有 NVSwitch）設計
  - 精確對齊驅動版本（以 **595** 版本為例）。
* 步驟 2 有針對 gpu 不同模式設計的三種方案，擇一使用

---
# 步驟

## 1.0 前處理
### 1.1 更新系統套件源
sudo apt update && sudo apt upgrade -y

### 1.2 安裝 dkms 與編譯核心工具（確保驅動與內核升級後能自動編譯）
sudo apt install -y dkms build-essential pahole

### 1.3 安裝特定版本的 NVIDIA Open 驅動 (鎖定 595 版本)
sudo apt install -y nvidia-driver-595-open nvidia-utils-595

### 1.4 鎖定驅動版本，防止日後 apt upgrade 自動升級搞壞環境
sudo apt-mark hold nvidia-driver-595-open nvidia-utils-595


## 2.0 重開機（必要）
sudo reboot
> ⚠️ **重要：**
> * **[方案 A] 單GPU VM**，核心動作是「關閉預設功能（拆東西）」，主動告訴驅動「我只有自己，別去管什麼 NVLink 織網，否則報出 `Error 802: system not yet initialized`。
> * **[方案 B] 多GPU VM**，繞過 VM 內部對 Fabric Manager 的無效聯絡，但**保留 NVLink P2P 通訊**，使多卡在虛擬機內仍能享受數百 GB/s 的高速互連！
> * **[方案 C] HGX實體機**，核心動作則是「加裝管理員軟體」，安裝 nvidia-fabricmanager 來管理多卡間的高速通訊

#### A_ VM w/ 單卡 ####
### 2.A.1 建立驅動設定檔，禁用 NVLink 與 Fabric 註冊 (這步驟如果用了 2.B.1 的做法，也會報錯 Error 802: system not yet initialized)
echo -e 'options nvidia NVreg_RegistryDwords="RMConnectToFabric=0"\noptions nvidia NVreg_NvLinkDisable=1' | sudo tee /etc/modprobe.d/nvidia.conf

### 2.A.2 卸載並重新載入驅動模組（或直接 sudo reboot 重啟系統）
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
sudo modprobe nvidia && sudo modprobe nvidia_uvm

### 2.A.3 檢查 GPU 狀態與 Fabric 狀態 (State 欄位此時應轉為 N/A)
nvidia-smi
nvidia-smi -q | grep -A 10 "Fabric"
> State 應成功轉為 N/A

#### B_ VM w/ 多卡 ####
### 2.B.1 建立驅動設定檔， 此處「不可」加入 NVreg_NvLinkDisable=1，否則多卡間的 NVLink 通訊會被切斷！
echo 'options nvidia NVreg_RegistryDwords="RMConnectToFabric=0"' | sudo tee /etc/modprobe.d/nvidia.conf

### 2.B.2 卸載並重新載入驅動模組（或直接 sudo reboot 重啟系統）
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
sudo modprobe nvidia && sudo modprobe nvidia_uvm

### 2.B.3 檢查 GPU 狀態與 Fabric 狀態 (State 欄位此時應轉為 N/A)
nvidia-smi
nvidia-smi -q | grep -A 10 "Fabric"
> (State 應轉為 N/A，但 nvlink 依舊保持啟用)

#### C_ 實體機 w/ 多卡 ####
### 2.C.1 安裝與驅動對齊版本的 Fabric Manager (鎖定 595 版本)
sudo apt-get install -y nvidia-fabricmanager-595

### 2.C.2 鎖定 Fabric Manager 版本防止更新
sudo apt-mark hold nvidia-fabricmanager-595

### 2.C.3 設定開機自動啟動並立即啟動
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager

### 2.C.4 確認服務正常運行
sudo systemctl status nvidia-fabricmanager
> （Active 應為 active (running)）

### 2.C.5 檢查 Fabric 織網狀態 (State 應轉為 Completed)
nvidia-smi -q | grep -A 10 "Fabric"
> (State 應轉為 Completed)

## 3.0 
### 3.1 cuda - 加入 NVIDIA 官方 repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

> 安裝最新 CUDA 13.3（對應 H200）(2026.6官宣)
sudo apt install -y cuda-toolkit-13-3

### 3.2 設定環境變數 (避免重複寫入)
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
source ~/.bashrc

### 3.3 確認 CUDA 版本
nvcc --version

## 4.0 conda 測試
### 4.1 下載 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

### 4.2 安裝（全程按 Enter / yes）
bash ~/miniconda.sh -b -p $HOME/miniconda3
rm ~/miniconda.sh


### 4.3 啟動 conda (避免重複寫入)
if ! grep -q "conda shell.bash hook" ~/.bashrc; then
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
fi
source ~/.bashrc

### 4.4 同意conda 授權
conda --version

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

### 4.5 切換至 conda 測試環境並驗證 
> * PyTorch 官方安裝包目前支持 cu121/cu124，能完美向前相容於 CUDA 13.x 驅動
conda create -n pytorch-env python=3.11 -y 
conda activate pytorch-env

> * 好處是百分百安裝到的 gpu 加速版，且 cuda 12.4 對 h200支援更好
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cat << 'EOF' > ~/test_gpu.py
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
EOF

python3 ~/test_gpu.py
> * 預期2秒內產生執行結束報告

## 5.0 安裝ib 卡 與 nvme 
### 5.1 檢查
lsmod | grep -E "mlx5|ib_"  # ib卡 (或用 lspci 來看)
lsblk   # nvme

### 5.2 安裝套件
sudo apt-get install -y linux-modules-extra-$(uname -r) rdma-core ibverbs-providers ibverbs-utils ibutils nvme-cli

### 5.3 手動載入 InfiniBand 驅動模組（或直接重開機自動載入）
sudo modprobe mlx5_ib

### 5.4 驗證硬體上線狀態
> (1) 檢查 ConnectX-7 網卡，狀態應顯示
ibv_devinfo
>  PORT_ACTIVE (4)

> (2) 檢查直通的 NVMe 本地高速 SSD 設備列表
sudo nvme list

## 6.0 (opt) 格式化與掛載 NVMe 本地硬碟至 /data
> 說明：清除可能殘留的舊 RAID 標籤，格式化為 ext4，並建立掛載點
sudo wipefs -a /dev/nvme0n1
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /data
sudo mount /dev/nvme0n1 /data

> 寫入 /etc/fstab 以便日後開機自動掛載（加上 nofail 參數防止硬體變動導致開機卡死）
echo '/dev/nvme0n1 /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab

## 7.0 (opt) ib 卡掛載給 ip
> 1. 動態取得 ens2 主要網卡 IP 的最後一碼 (本機 192.168.230.88 ➡️ 取得 88)
LAST_OCTET=$(ip -4 addr show ens2 | grep -oP 'inet \K[\d.]+' | cut -d. -f4)

> 2. 自動偵測本機 InfiniBand 網卡的介面名稱 (精確抓取通訊介面如 ibs5 或 ib0)
IB_DEV=$(ls /sys/class/net | grep -E "^ib" | head -n 1)

> 3. 為 InfiniBand 配置 10.0.0.<ip> 的獨立高速網段並啟用
sudo ip addr add 10.0.0.${LAST_OCTET}/24 dev $IB_DEV
sudo ip link set dev $IB_DEV up

> 驗證配置結果
ip addr show dev $IB_DEV
df -h /data


## 8.0 (opt)執行檢查腳本
./sanity_check.sh

