import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

rgbs = pd.read_csv("data_lucas_hsv_xy.csv").to_numpy()
x = rgbs[:,0]
y = rgbs[:,1]
plt.grid()
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.scatter(0, 0, color="red")
plt.annotate('(0,0)', xy =(0, 0), )
plt.scatter(x,y)
plt.show()