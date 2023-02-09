from ds_manager import DSManager
from scipy.stats import pearsonr
import os
os.chdir("../")
from sklearn.linear_model import LinearRegression


def r2_xy(x,y):
    model_instance = LinearRegression()
    model_instance = model_instance.fit(x,y)
    return model_instance.score(x,y)

def calculate_r2(ds:DSManager):
    x = ds.full_ds.get_x()
    y = ds.full_ds.get_y()
    y = y.reshape(-1,1)

    r1 = r2_xy(x[:,0:1], y)
    r2 = r2_xy(x[:,1:2], y)
    r3 = r2_xy(x[:,2:3], y)

    return r1, r2, r3


rgb = DSManager("lucas", "rgb")
hsv = DSManager("lucas", "hsv")
xyz = DSManager("lucas", "xyz")
xyy = DSManager("lucas", "xyy")
cielab = DSManager("lucas", "cielab")

print(calculate_r2(rgb))
print(calculate_r2(hsv))
print(calculate_r2(xyz))
print(calculate_r2(xyy))
print(calculate_r2(cielab))

