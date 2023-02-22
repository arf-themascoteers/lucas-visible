import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import ds_manager
import os
os.chdir("../")
import torch
from train import train
from test import test

#x = 0
def predict(dm):
    # global x
    # x = x+1
    # return None,None, x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = train(device, dm.train_ds, nn_config={"num_epochs":100})
    torch.save(model_instance,"soc.h5")
    r2, y_hat = test(device, dm.test_ds, model_instance, return_pred=True)
    y = dm.test_ds.get_y()
    return y, y_hat, r2


dss = ["lucas","raca","ossl"]
css = ["red","green","blue","hue","saturation","value","l","a","b"]

ar = np.zeros((len(dss),len(css)))
for ds_index, ds in enumerate(dss):
    for cs_index, cs in enumerate(css):
        dm = ds_manager.DSManager(ds, cs)
        y, y_hat, r2 = predict(dm)
        print(f"{cs} {ds} {r2}")
        ar[ds_index,cs_index] = np.round(r2,3)

means = np.round(np.mean(ar, axis=0, keepdims=True))
ar = np.concatenate((ar,means), axis=0)

df = pd.DataFrame(data=ar, columns=css, index=dss+["mean"])
df.to_csv("nn_impact2.csv")




