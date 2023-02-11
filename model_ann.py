import torch.nn as nn
import torch
import torch.nn.functional as F
## normal
## with Relu(-weight)


class ANN(nn.Module):
    def __init__(self, size=3, mid=50):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(size, mid),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(size, 5),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(5, 1)
        )

        self.fc1_w = nn.Parameter(torch.tensor(0.5))

        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

    def calculate_loss(self, x1, x2, y):
        x1 = x1.reshape(-1)
        x2 = x2.reshape(-1)
        x1 = self.criterion(x1, y)
        x2 = self.criterion(x2, y)
        loss = (x1 * self.fc1_w + x2 * (1-self.fc1_w))**2
        return loss

    def predict(self, x):
        x1, x2 = self.forward(x)
        return (x1 * self.fc1_w + x2 * (1-self.fc1_w))
