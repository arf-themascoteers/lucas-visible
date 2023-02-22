import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import ds_manager
import os
os.chdir("../")

ar = pd.read_csv("nn_impact3.csv").to_numpy()
ar = ar[:,-1]
for i in range(0,9,3):
    thesum = np.sum(ar[i:i+3])
    ar[i:i+3] = ar[i:i+3]/thesum
print(ar)
