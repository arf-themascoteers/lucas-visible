from sklearn.linear_model import LinearRegression
import numpy as np
import ds_manager
from scipy.stats import pearsonr
import os
os.chdir("../")
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch
from train import train
from test import test

def predict(dm):
    # global x
    # x = x+1
    # return None,None, x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_instance = train(device, dm.train_ds, nn_config={"num_epochs":100})
    # torch.save(model_instance,"soc.h5")
    model_instance = torch.load("soc.h5")
    r2, y_hat = test(device, dm.test_ds, model_instance, return_pred=True, shuffle=False)
    y = dm.test_ds.get_y()
    return y, y_hat, r2

def draw(dm, ax):
    y,y_hat,r2 = predict(dm)
    y = y[0:1000]
    y_hat = y_hat[0:1000]
    model_instance = LinearRegression()
    model_instance = model_instance.fit(y_hat.reshape(-1,1),y)
    print(model_instance.score(y_hat.reshape(-1,1),y))
    coef = model_instance.coef_[0]
    intercept = model_instance.intercept_
    x_vals = np.linspace(np.min(y_hat),np.max(y_hat),100)
    y_vals = x_vals * coef + intercept
    ax.set_xlim([0,0.2])
    ax.set_ylim([0,0.2])
    ax.scatter(y_hat,y, s=1)
    ax.plot(x_vals, y_vals, "g")
    return r2


dss = ["lucas","raca","ossl"]
dss = ["lucas"]
css = ["red","green","blue","hue","saturation","value","l","a","b"]
css = ["hue"]
fig, ax = plt.subplots(1, 1)
fig.tight_layout(pad=1.0)
ar = np.zeros((len(dss),len(css)))

for ds_index, ds in enumerate(dss):
    for cs_index, cs in enumerate(css):
        dm = ds_manager.DSManager(ds, cs)
        r2 = draw(dm, ax)
        print(f"{cs} {ds} {r2}")
        plt.show()
        exit()
        ar[ds_index,cs_index] = np.round(r2,3)

means = np.mean(ar, axis=0, keepdims=True)
ar = np.concatenate((ar,means), axis=0)

df = pd.DataFrame(data=ar, columns=css, index=dss+["mean"])
df.to_csv("ind_impact.csv")
