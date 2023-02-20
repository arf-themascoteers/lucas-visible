import pandas as pd
import colour
import numpy as np


def process(inf, outf):
    df = pd.read_csv(inf)
    npdf = df.to_numpy()
    target_npdf = np.zeros((npdf.shape[0],2))
    target_npdf[:,0] = npdf[:,0]
    target_npdf[:,1] = npdf[:,-1]
    df = pd.DataFrame(target_npdf, columns = ['hue','oc'])
    df.to_csv(outf, index=False)
    print("done")
