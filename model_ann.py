import torch.nn as nn
## 100X5X1 -> 0.710
## 100X50X1 -> 0.727
## 50X10X5X1 -> 0.722
## 50X5X1 -> 0.726
## 50X1 -> 0.721
## 50XdropoutX1 ->
## 50X5XdropoutX1 -> 0.688
## 50 BN 5 1 -> 0.717
## 50 5 1 -> 726
## 100 20 1 -> 727
## 100 20 1 - (bs 50) -> 717
## 100 20 1 - (Epoch 1000) -> 732
## 100 20 1 - LR
## 100 20 1 - Epoch


class ANN(nn.Module):
    def __init__(self, size=3, mid=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, mid),
            nn.LeakyReLU(),
            nn.Linear(mid,20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

