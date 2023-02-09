import pandas as pd
import numpy as np
import os
os.chdir("../")

npdf = pd.read_csv("impacts3.csv").to_numpy()
npdf = np.round(npdf,3)
df = pd.DataFrame(data=npdf)
df.to_csv("impact4.csv",index=False)