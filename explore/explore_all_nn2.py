import torch
from train import train
from test import test
import numpy as np
import ds_manager
import os
os.chdir("../")
import pandas as pd


def calculate_grads(dm:ds_manager.DSManager, c):
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = train(device, train_ds)
    torch.save(model_instance,f"{c}.h5")

    x = test_ds.get_x()
    x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)
    y = test_ds.get_y()
    y = torch.tensor(y, dtype=torch.float32)
    y = y.to(device)
    y = y.reshape(-1, 1)
    x.requires_grad = True
    criterion = torch.nn.MSELoss(reduction='mean')
    y_hat = model_instance(x)
    loss = criterion(y_hat, y)
    loss.backward()
    grads = torch.abs(x.grad).sum(axis=0)
    grads = (grads / torch.sum(grads))
    return grads[0].item(), grads[1].item(), grads[2].item()


css = ["rgb","hsv","cielab"]
dss = ["lucas", "raca", "ossl"]
filename = "impacts2.csv"
results = np.zeros((9,3))
for cspace, c in enumerate(css):
    for col, ds in enumerate(dss):
        dm = ds_manager.DSManager(ds, c)
        i0, i1, i2 = calculate_grads(dm, c)
        start_index = cspace*3
        results[start_index, col] = i0
        results[start_index + 1, col] = i1
        results[start_index + 2, col] = i2

means = np.mean(results, axis=1, keepdims=True)
results = np.round(np.concatenate((results,means), axis=1),2)
df = pd.DataFrame(results, columns=dss+["mean"])
df.to_csv(filename, index=False)
