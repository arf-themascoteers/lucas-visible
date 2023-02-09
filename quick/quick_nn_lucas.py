import os
os.chdir("../")
import torch
from train import train
from test import test
import ds_manager
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def calculate_r2(dm, model):
    train_ds = dm.train_ds
    test_ds = dm.test_ds
    if model == "ann":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_instance = train(device, train_ds)
        return test(device, test_ds, model_instance)
    else:
        model_instance = RandomForestRegressor(max_depth=5, n_estimators=100)
        model_instance = model_instance.fit(train_ds.get_x(), train_ds.get_y())
        return model_instance.score(test_ds.get_x(), test_ds.get_y())

for model in ["rf", "ann"]:
    print(model)
    print("===========")
    sis = ds_manager.DSManager.si_list()
    for x in range(0, len(sis)):
        i = sis[x]
        print(i, calculate_r2(ds_manager.DSManager("lucas", "hsv", si=[i], si_only=True), model))
    print("Only SI", calculate_r2(ds_manager.DSManager("lucas", "hsv", si=sis, si_only=True), model))
    print("hsv-all-si",calculate_r2(ds_manager.DSManager("lucas", "hsv", si=sis), model))
    print("hsv",calculate_r2(ds_manager.DSManager("lucas", "hsv"), model))


