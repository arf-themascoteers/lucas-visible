import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

os.chdir("../")
rgbs = pd.read_csv("data/lucas/rgb.csv").to_numpy()

fig, ax = plt.subplots(1,1)

soc = rgbs[:,3]
soc = np.log(soc)
counts, bins = np.histogram(soc, bins=50)
ax.hist(bins[:-1], bins, weights=counts)
ax.set_title('SOC')



fig.tight_layout(pad=1.0)

plt.show()
