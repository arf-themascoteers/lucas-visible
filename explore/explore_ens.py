import ds_manager
import os
os.chdir("../")
from model_ann import ANN
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = ds_manager.DSManager("lucas", cspace="hsv").test_ds
model = torch.load("ann.h5")
print("weight",model.fc1_w.item())

batch_size = 1
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
model.eval()
model.to(device)

for (x, y) in dataloader:
    x = x.to(device)
    y = y.to(device)
    print("actual y",y.item())
    y_hat = model(x)
    print("predicted y",y_hat.item())
    exit(0)


