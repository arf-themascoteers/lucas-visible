import pandas as pd
import colour
import numpy as np


df = pd.read_csv("data_rgb_absorbance.csv")
npdf = df.to_numpy()
npdf[:, 0:3] = 1 / (10 ** npdf[:, 0:3])
#npdf[:, 0:3] = npdf[:, 0:3] / np.max(npdf[:, 0:3])
print(npdf[:, 0].min())
print(npdf[:, 1].min())
print(npdf[:, 2].min())
print(npdf[:, 0].max())
print(npdf[:, 1].max())
print(npdf[:, 2].max())
df = pd.DataFrame(npdf, columns = ['r','g','b','oc'])
df.to_csv("data_rgb.csv", index=False)
print("done")