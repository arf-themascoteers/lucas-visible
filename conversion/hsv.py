import pandas as pd
import colour
import numpy as np


def process(inf, outf):
    df = pd.read_csv(inf)
    npdf = df.to_numpy()
    for i in range(len(npdf)):
        npdf[i,0:3] = colour.RGB_to_HSV(npdf[i,0:3])
    df = pd.DataFrame(npdf, columns = ['h','s','v','oc'])
    df.to_csv(outf, index=False)
    print("done")
