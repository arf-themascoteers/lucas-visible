import pandas as pd
import numpy as np
import math


def get_x_y(h):
    angle = 2 * math.pi * h
    x = math.cos(angle)
    y = math.sin(angle)
    return x,y


def process(inf, outf):
    df = pd.read_csv(inf)
    npdf = df.to_numpy()
    mydata = np.zeros((npdf.shape[0],npdf.shape[1]+1))
    mydata[:,2:] = npdf[:,1:]
    for i in range(len(npdf)):
        x,y = get_x_y(npdf[i,0])
        mydata[i,0] = x
        mydata[i,1] = y

    df = pd.DataFrame(mydata, columns = ['HX', 'HY', 'S','V','oc'])
    df.to_csv(outf, index=False)
    print("done")



if __name__ == "__main__":
    process("data_lucas_rgb.csv", "data_lucas_hsv_xy.csv")
