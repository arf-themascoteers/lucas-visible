from sklearn.linear_model import LinearRegression
import numpy as np
import ds_manager
from scipy.stats import pearsonr
import os
os.chdir("../")
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def pearson(dm):
    r = pearsonr(dm.full_ds.get_x().reshape(-1), dm.full_ds.get_y())
    return r.statistic


dss = ["lucas","raca","ossl"]
css = ["red","green","blue","hue","saturation","value","l","a","b"]

ar = np.zeros((len(dss),len(css)))

for ds_index, ds in enumerate(dss):
    for cs_index, cs in enumerate(css):
        dm = ds_manager.DSManager(ds, cs)
        r = pearson(dm)
        print(f"{cs} {ds} {r}")
        ar[ds_index,cs_index] = np.round(r,3)

means = np.mean(ar, axis=0, keepdims=True)
ar = np.concatenate((ar,means), axis=0)

df = pd.DataFrame(data=ar, columns=css, index=dss+["mean"])
df.to_csv("linear_r.csv")
