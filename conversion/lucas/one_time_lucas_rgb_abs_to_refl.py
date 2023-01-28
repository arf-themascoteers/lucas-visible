import pandas as pd
import os


def process():
    basedir = f"data/lucas"
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    df = pd.read_csv("data/lucas_rgb_absorbance.csv")
    npdf = df.to_numpy()
    npdf[:, 0:3] = 1 / (10 ** npdf[:, 0:3])
    df = pd.DataFrame(npdf, columns = ['r','g','b','oc'])
    df.to_csv(f"{basedir}/rgb.csv", index=False)
    print("done")


if __name__ == "__main__":
    process()