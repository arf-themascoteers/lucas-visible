import pandas as pd
import os


def process():
    basedir = f"data/lucas"
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    df = pd.read_csv("data/lucas_rgb_absorbance.csv")
    npdf = df.to_numpy()
    npdf[:, 0:3] = 1 / (10 ** npdf[:, 0:3])
    #npdf[:, 0:3] = npdf[:, 0:3] / np.max(npdf[:, 0:3])
    print(npdf[:, 0].min())
    print(npdf[:, 1].min())
    print(npdf[:, 2].min())
    print(npdf[:, 0].max())
    print(npdf[:, 1].max())
    print(npdf[:, 2].max())
    df = pd.DataFrame(npdf, columns = ['r','g','b','oc'])
    df.to_csv(f"{basedir}/rgb.csv", index=False)
    print("done")


if __name__ == "__main__":
    process()