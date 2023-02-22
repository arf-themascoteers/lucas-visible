from sklearn.linear_model import LinearRegression
import numpy as np
import ds_manager
import os
os.chdir("../")
import pandas as pd
from sklearn.metrics import r2_score




dss = ["lucas", "raca", "ossl"]


for ds_index, ds in enumerate(dss):
    dm = ds_manager.DSManager(ds, "rgb")
    X = dm.full_ds.get_x()
    counts = np.argmax(X,axis=1)
    red = np.count_nonzero(counts == 0)
    green = np.count_nonzero(counts == 1)
    blue = np.count_nonzero(counts == 2)
    print(dss)
    print(red,green,blue)


