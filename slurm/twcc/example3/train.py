import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import argparse
import os
import time

print("before running dist.init_process_group()")
MASTER_ADDR = os.environ["MASTER_ADDR"]
MASTER_PORT = os.environ["MASTER_PORT"]
LOCAL_RANK = os.environ["LOCAL_RANK"]
RANK = os.environ["RANK"]
WORLD_SIZE = os.environ["WORLD_SIZE"]

print("MASTER_ADDR: {}\tMASTER_PORT: {}".format(MASTER_ADDR, MASTER_PORT))
print("LOCAL_RANK: {}\tRANK: {}\tWORLD_SIZE: {}".format(LOCAL_RANK, RANK, WORLD_SIZE))


# 設定參數
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='訓練批次大小')
parser.add_argument('--epochs', type=int, default=10, help='訓練回合數')
parser.add_argument('--lr', type=float, default=0.01, help='學習率')
args = parser.parse_args()
print("1_start")
# 初始化 DDP 環境
dist.init_process_group(backend='nccl')

# 計算每個節點上的 GPU 編號
#torch.cuda.set_device(f"cuda:{RANK}") # 再多節點這樣宣告會出錯，修改成以下
torch.cuda.set_device(f"cuda:{LOCAL_RANK}")


# 載入和正規化 mnist 資料集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

print("2_download")
# 使用 torchvision.datasets.MNIST 的 download 參數來自動下載資料集
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

print("3_net")
# 定義一個卷積神經網路
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.LogSoftmax(dim=1)(x)
        return output

# 建立模型並且設定為 DDP 模式
model = Net().cuda()
#model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
model = nn.parallel.DistributedDataParallel(model)

# 定義損失函數和優化器
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# 訓練模型
def train(epoch):
    model.train()
    print("4.1 train()")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 取得輸入
        inputs, labels = data[0].cuda(), data[1].cuda()
        # 歸零梯度
        optimizer.zero_grad()
        # 前向傳播 + 反向傳播 + 優化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 累積損失
        running_loss += loss.item()
        # 每 2000 批次列印統計資訊
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# 測試模型
def test():
    print("4.2 test()")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

print("4_run")
# 執行訓練和測試
for epoch in range(args.epochs):
    train(epoch)
    test()
