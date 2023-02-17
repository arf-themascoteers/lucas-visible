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
# 200 1 - (Epoch 300) -> 708
# 200 1 - (LR 0.001) -> 0.56
# 200 1 - (LR 0.1) -> 0.56
# 200 1 - (LR 0.01) -> 0.56
##=======================================
# 3 50 1  ->
# 3 50 3 1  ->
# 3 200 3 1  ->
# 3 200 10 1  ->
# 3 20 10 1  ->
# 3 50 10 1  ->
# 200 1  ->
# 300 1 -  ->
# 400 1 -  ->
# 400 1 -  ->
# 200 5 1 -  ->
## 100 20 1 - LR
## 100 20 1 - Epoch
## 100 1 - Epoch
## 100 1 (log-normalize) - Epoch


class ANN(nn.Module):
    def __init__(self, size=3, mid=50):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, mid),
            nn.LeakyReLU(),
            nn.Linear(mid, 1),
            # nn.LeakyReLU(),
            # nn.Linear(mid, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

