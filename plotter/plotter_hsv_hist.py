import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

os.chdir("../")

# rgbs = pd.read_csv(f"data/ossl_rgb_raw.csv").to_numpy()
# print(f"r: {np.round(np.min(rgbs[:, 0]), 3)}\t{np.round(np.max(rgbs[:, 0]), 3)}\t{np.round(np.mean(rgbs[:, 0]), 3)}")
# print(f"g: {np.round(np.min(rgbs[:, 1]), 3)}\t{np.round(np.max(rgbs[:, 1]), 3)}\t{np.round(np.mean(rgbs[:, 1]), 3)}")
# print(f"b: {np.round(np.min(rgbs[:, 2]), 3)}\t{np.round(np.max(rgbs[:, 2]), 3)}\t{np.round(np.mean(rgbs[:, 2]), 3)}")
#
# exit()

# for ds in ["lucas", "raca", "ossl"]:
#     for cs in ["rgb", "hsv", "cielab"]:
#         rgbs = pd.read_csv(f"data/{ds}/{cs}.csv").to_numpy()
#         print(f"{ds} {cs}")
#         print(f"1: {np.round(np.min(rgbs[:,0]),3)}\t{np.round(np.max(rgbs[:,0]),3)}\t{np.round(np.mean(rgbs[:,0]),3)}")
#         print(f"2: {np.round(np.min(rgbs[:,1]),3)}\t{np.round(np.max(rgbs[:,1]),3)}\t{np.round(np.mean(rgbs[:,1]),3)}")
#         print(f"3: {np.round(np.min(rgbs[:,2]),3)}\t{np.round(np.max(rgbs[:,2]),3)}\t{np.round(np.mean(rgbs[:,2]),3)}")
#
#
# exit(0)

rgbs = pd.read_csv("data/ossl/hsv.csv").to_numpy()

#plt.xlim(0,1)
fig, ax = plt.subplots(1,3)

hue = rgbs[:,0]
counts, bins = np.histogram(hue)
ax[0].hist(bins[:-1], bins, weights=counts)
ax[0].set_title('Hue')
ax[0].set_xlim([0, 1])

saturation = rgbs[:,1]
counts, bins = np.histogram(saturation)
ax[1].hist(bins[:-1], bins, weights=counts)
ax[1].set_title('Saturation')
ax[1].set_xlim([0, 1])

value = rgbs[:,2]
counts, bins = np.histogram(value)
ax[2].hist(bins[:-1], bins, weights=counts)
ax[2].set_title('Value')
ax[2].set_xlim([0, 1])

fig.tight_layout(pad=1.0)


plt.show()
