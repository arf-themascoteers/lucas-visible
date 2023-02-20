import pandas as pd
import numpy as np
import os
os.chdir("../../")


def process():
    basedir = f"data/ossl"
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    df = pd.read_csv("data/ossl_rgb_raw.csv")
    npdf = df.to_numpy()
    npdf[:, 0:3] = npdf[:, 0:3] / np.max(npdf[:, 0:3], axis = 0)
    df.to_csv(f"{basedir}/rgb.csv", index=False)
    print("done")


if __name__ == "__main__":
    process()