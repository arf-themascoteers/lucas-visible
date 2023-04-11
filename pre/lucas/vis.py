import os

import numpy as np
import pandas as pd


def vis():
    src_file = "data/out/full.csv"
    out_file = "data/out/vis.csv"
    src_df = pd.read_csv(src_file)

    src_df = src_df[["665","560","490","phc","phh","ec","caco3","p","n","k","oc"]]
    src_df.to_csv(out_file,index=False)
    print("done")


if __name__ == "__main__":
    vis()