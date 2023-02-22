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
    print(model_instance.score(c, y))
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


dm = ds_manager.DSManager("lucas", "hue", normalize=False)
draw(dm.full_data)