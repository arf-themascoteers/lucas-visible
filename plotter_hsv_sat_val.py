import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

rgbs = pd.read_csv("data_lucas_hsv_xy.csv").to_numpy()
sat = rgbs[:,2]
counts, bins = np.histogram(sat)
print(counts)
print(bins)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

val = rgbs[:,3]
counts, bins = np.histogram(val)
print(counts)
print(bins)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

