import os

import numpy as np
import pandas as pd

src_file = "data/out/full.csv"
out_file = "data/out/vis.csv"
src_df = pd.read_csv(src_file)

src_df = src_df[["665","560","490","phh","oc"]]
src_df.to_csv(out_file,index=False)
print("done")

