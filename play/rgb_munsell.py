import colour
import os
os.chdir("../")
import pandas as pd
import numpy as np

# ar = pd.read_csv("data/lucas/rgb.csv").to_numpy()[:,0:3]
# m = np.max(ar)
# ar = ar/m
#for rgb in ar:
rgb = [1,0,.1]
XYZ = colour.sRGB_to_XYZ(rgb)
xyY = colour.XYZ_to_xyY(XYZ)
munsell = colour.xyY_to_munsell_colour(xyY)
print(munsell)

