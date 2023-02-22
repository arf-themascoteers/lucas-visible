import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

os.chdir("../")
fig, ax = plt.subplots(1, 3)
names = ["Lucas", "RaCA", "OSSL"]
for index,d in enumerate(["lucas", "raca", "ossl"]):
    rgbs = pd.read_csv(f"data/{d}/hsv.csv").to_numpy()
    hue = rgbs[:,0]
    counts, bins = np.histogram(hue, bins=200)
    ax[index].hist(bins[:-1], bins, weights=counts)
    ax[index].set_title(names[index])
    ax[index].set_xlim([0, 1])
    fig.tight_layout(pad=1.0)

plt.show()
