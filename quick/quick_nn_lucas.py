import os
os.chdir("../")
import torch
from train import train
from test import test
import ds_manager
import numpy as np


def calculate_r2(dm):
    train_ds = dm.train_ds
    test_ds = dm.test_ds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = train(device, train_ds)
    return test(device, test_ds, model_instance)

sis = ds_manager.DSManager.si_list()
for x in range(0,len(sis)):
    i = sis[x]
    print(i,calculate_r2(ds_manager.DSManager("lucas", "hsv", si=[i], si_only=True)))

print("hsv",calculate_r2(ds_manager.DSManager("lucas", "hsv")))
print("hsv-all-si",calculate_r2(ds_manager.DSManager("lucas", "hsv", si=sis)))


# print(calculate_r2(ds_manager.DSManager("lucas","hsv", si=["ibs"], si_only=True)))
# print(calculate_r2(ds_manager.DSManager("lucas","hsv", si=["soci"], si_only=True)))
# print(calculate_r2(ds_manager.DSManager("lucas","hsv", si=["soci", "ibs"], si_only=True)))
# print(calculate_r2(ds_manager.DSManager("lucas","hsv", si=["soci", "ibs"])))
# print(calculate_r2(ds_manager.DSManager("lucas","hsv", si=["soci"])))
# print(calculate_r2(ds_manager.DSManager("lucas","hsv", si=["ibs"])))
# print(calculate_r2(ds_manager.DSManager("lucas","hsv")))


