import pandas as pd
import colour
import numpy as np


df = pd.read_csv("data_rgb.csv")
npdf = df.to_numpy()
for i in range(len(npdf)):
    npdf[i,0:3] = colour.sRGB_to_XYZ(npdf[i,0:3])
print("Original")
print(npdf[:, 0].min())
print(npdf[:, 1].min())
print(npdf[:, 2].min())
print(npdf[:, 0].max())
print(npdf[:, 1].max())
print(npdf[:, 2].max())
df = pd.DataFrame(npdf, columns = ['X','Y','Z','oc'])
df.to_csv("data_XYZ_original.csv", index=False)

max_values = np.max(npdf[:, 0:3])
npdf[:, 0:3] = npdf[:, 0:3] / max_values
print("Normalized")
print(npdf[:, 0].min())
print(npdf[:, 1].min())
print(npdf[:, 2].min())
print(npdf[:, 0].max())
print(npdf[:, 1].max())
print(npdf[:, 2].max())
df = pd.DataFrame(npdf, columns = ['X','Y','Z','oc'])
df.to_csv("data_XYZ.csv", index=False)


print("done")