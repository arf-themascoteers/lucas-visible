import os
os.chdir("../")
import torch
from train import train
from test import test
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ds_manager import DSManager
import time

start = time.time()
dss = DSManager("lucas", "hsv")
train_ds = dss.get_train_ds()
test_ds = dss.get_test_ds()
model_instance = RandomForestRegressor(max_depth=4, n_estimators=1000)
model_instance = model_instance.fit(train_ds.get_x(), train_ds.get_y())
s = model_instance.score(test_ds.get_x(), test_ds.get_y())
print(s)
end = time.time()
print(f"Required {end-start}")
