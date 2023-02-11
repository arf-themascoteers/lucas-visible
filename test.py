import torch
from spectral_dataset import SpectralDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import time


def test(device, ds, model):
    batch_size = 30000
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.eval()
    model.to(device)

    r2 = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model.predict(x)
        r2 = r2_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        print(f"R^2 {r2:.4f}")

    return r2

