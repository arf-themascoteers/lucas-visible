from sklearn.linear_model import LinearRegression
import ds_manager
from scipy.stats import pearsonr
import os
os.chdir("../")
import matplotlib.pyplot as plt
import numpy as np
import torch
from train import train
from test import test


def predict(dm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_instance = torch.load("soc.h5")
    model_instance = train(device, dm.train_ds, nn_config={"num_epochs":100})
    torch.save(model_instance,"soc.h5")
    r2, y_hat = test(device, dm.test_ds, model_instance, return_pred=True)
    y = dm.test_ds.get_y()
    return y, y_hat, r2


def plot_comp(dm, ax):
    y,y_hat,r2= predict(dm)
    y = y[0:500]
    y_hat = y_hat[0:500]
    print(r2)
    model = LinearRegression()
    model = model.fit(y_hat.reshape(-1,1), y)
    coef = model.coef_[0]
    inter = model.intercept_
    x_val = np.linspace(0,1,1000)
    y_val = x_val * coef + inter
    ax.scatter(y, y_hat, s=1, c="b")
    ax.plot(x_val, y_val, "g")


def draw(dm):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 0.1])
    plot_comp(dm, ax)

    fig.tight_layout(pad=1.0)
    plt.show()

# x = np.linspace(0,1, 100)
# y = x *.2 +.1
#
# plt.scatter(x,y, c='b', s=1)
# plt.show()
# exit(0)

for cs in ["hue"]:
    for ds in ["lucas"]:
        dm = ds_manager.DSManager(ds, cs)
        draw(dm)


