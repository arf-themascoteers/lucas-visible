import numpy as np
import pandas as pd
import os

os.chdir("../")
df = pd.read_csv(f"data/lucas/rgb.csv")
npdf = df.to_numpy()
maxes = np.round(np.max(npdf, axis=0),3)
mins = np.round(np.min(npdf, axis=0),3)
print(maxes)
print(mins)
