import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self, size=3, mid=50):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, mid),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(mid),
            nn.Linear(mid, 10),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(10),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

