from sklearn.linear_model import LinearRegression
import numpy as np
import ds_manager
import os
os.chdir("../")
import pandas as pd
from sklearn.metrics import r2_score


def predict(dm):
    model_instance = LinearRegression().fit(dm.full_ds.get_x(), dm.full_ds.get_y())
    y_hat = model_instance.predict(dm.full_ds.get_x())
    return dm.full_ds.get_y(), y_hat, r2_score(dm.full_ds.get_y(), y_hat)


dss = ["lucas","raca","ossl"]
css = ["red","green","blue","hue","saturation","value","l","a","b"]

ar = np.zeros((len(css),len(dss)))

for ds_index, ds in enumerate(dss):
    for cs_index, cs in enumerate(css):
        dm = ds_manager.DSManager(ds, cs)
        y, y_hat, r2 = predict(dm)
        print(f"{cs} {ds} {r2}")
        ar[cs_index, ds_index] = r2

means = np.mean(ar, axis=1, keepdims=True)
ar = np.concatenate((ar,means), axis=1)
df = pd.DataFrame(data=ar, columns=dss+["mean"], index=css)
df.to_csv("linear_r2-original.csv")
ar = np.round(ar,2)

df = pd.DataFrame(data=ar, columns=dss+["mean"], index=css)
df.to_csv("linear_r2.csv")
