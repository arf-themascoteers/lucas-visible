import torch.nn as nn
## 100X5X1 -> 0.710
## 100X50X1 -> 0.727
## 50X10X5X1 -> 0.722
##


class ANN(nn.Module):
    def __init__(self, size=3, mid=50):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, mid),
            nn.LeakyReLU(),
            nn.Linear(mid,5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

