import os

import numpy as np
import pandas as pd


def vis():
    src_file = "data/out/full_with_empty.csv"
    out_file = "data/out/vis_with_empty.csv"
    src_df = pd.read_csv(src_file)

    src_df = src_df[["665","560","490","point_id","coarse","clay","sand","silt","phc","phh","ec","caco3","p","n","k"
        ,"elevation","lc1","lu1","stones","lon","lat","oc"]]
    src_df.to_csv(out_file,index=False)
    print("done")


if __name__ == "__main__":
    vis()