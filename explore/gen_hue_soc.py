from sklearn.linear_model import LinearRegression
import numpy as np
import ds_manager
from scipy.stats import pearsonr
import os
os.chdir("../")
import matplotlib.pyplot as plt
import numpy as np


def plot_comp(c, y, ax):
    model_instance = LinearRegression()
    model_instance = model_instance.fit(c, y)
    coef = model_instance.coef_[0]
    i = model_instance.intercept_
    c = c.reshape(-1)
    x_vals = np.linspace(np.min(c),np.max(c),100)
    y_vals = x_vals * coef + i
    ax.set_ylim(0,100)
    ax.scatter(c,y, s=1)
    ax.plot(x_vals, y_vals, "g")
    plt.xlabel('Hue')
    plt.ylabel('SOC')

def draw(data):
    fig, ax = plt.subplots(1, 1)
    # data = data[data[:,3]<50]
    data = data[0:1000,:]
    h = data[:,0:1]
    y = data[:,-1]
    plot_comp(h,y, ax)
    fig.tight_layout(pad=1.0)
    plt.show()


#["rgb", "hsv", "cielab"]
for cs in ["hue"]:
    r2_1s = []
    r2_2s = []
    r2_3s = []

    for ds in ["lucas"]:
        dm = ds_manager.DSManager(ds, cs, normalize=False)
        draw(dm.full_data)


    print(cs)

    print("=======r2-AVG=====")

    r2_1avg = np.round(sum(r2_1s) / len(r2_1s), 3)
    print(r2_1avg)
    r2_2avg = np.round(sum(r2_2s) / len(r2_2s), 3)
    print(r2_2avg)
    r2_3avg = np.round(sum(r2_3s) / len(r2_3s), 3)
    print(r2_3avg)
