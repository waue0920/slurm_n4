import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

import os
GPU_NUM=2  # 真實 GPU數量
os.environ["SLURM_NTASKS"] = "1"
os.environ["SLURM_NTASKS_PER_NODE"] = f"{GPU_NUM}" # 此數量需與 gpu 數對齊，但 slurm 的此數值設定為 1，故在此覆蓋

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def get_data():
    x = torch.randn(1000, 32)  # 
    y = torch.randint(0, 2, (1000,))  # 
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)

def main():
    model = SimpleModel()  # 
    data_loader = get_data()  #

    trainer = Trainer(
        devices=GPU_NUM,          # 
        accelerator="gpu",   # 
        strategy="ddp",      # 
        max_epochs=10,        # 
    )

    trainer.fit(model, data_loader)

if __name__ == "__main__":
    main()
