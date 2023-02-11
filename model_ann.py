import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, size=3, mid=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, mid),
            nn.LeakyReLU(),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

