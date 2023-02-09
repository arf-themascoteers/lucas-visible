from ds_manager import DSManager
from scipy.stats import pearsonr
import os
os.chdir("../")


def calculate_pearson(ds:DSManager):
    x = ds.full_ds.get_x()
    y = ds.full_ds.get_y()

    r1 = pearsonr(x[:,0], y)
    r2 = pearsonr(x[:,1], y)
    r3 = pearsonr(x[:,2], y)

    return r1.statistic, r2.statistic, r3.statistic


rgb = DSManager("lucas", "rgb")
hsv = DSManager("lucas", "hsv")
xyz = DSManager("lucas", "xyz")
xyy = DSManager("lucas", "xyy")
cielab = DSManager("lucas", "cielab")

print(calculate_pearson(rgb))
print(calculate_pearson(hsv))
print(calculate_pearson(xyz))
print(calculate_pearson(xyy))
print(calculate_pearson(cielab))

