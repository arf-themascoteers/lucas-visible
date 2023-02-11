import torch.nn as nn
import torch
import torch.nn.functional as F
## normal
## with Relu(-weight)


class ANN(nn.Module):
    def __init__(self, size=3, mid=50):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(size, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(size, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
        )

        self.fc1_w = nn.Parameter(torch.tensor(0.5))

        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return (x1 * self.fc1_w + x2 * (1-self.fc1_w))

    def calculate_loss(self, x, y):
        y_hat = self.forward(x)
        y_hat = y_hat.reshape(-1)
        loss = self.criterion(y, y_hat)
        return loss


