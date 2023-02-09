from ds_manager import DSManager
from scipy.stats import pearsonr
import os
os.chdir("../")
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.feature_selection import r_regression


def calculate_r2(ds:DSManager):
    x = ds.full_ds.get_x()
    y = ds.full_ds.get_y()

    return r_regression(x,y)


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

