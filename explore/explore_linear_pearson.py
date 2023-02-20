from sklearn.linear_model import LinearRegression
import numpy as np
import ds_manager
from scipy.stats import pearsonr
import os
os.chdir("../")

def calculate_r(train_ds, test_ds):
    train_x = train_ds.get_x()
    train_y = train_ds.get_y()
    test_x = test_ds.get_x()
    test_y = test_ds.get_y()

    r1 = pearsonr(train_x[:,0], train_y)
    r2 = pearsonr(train_x[:,1], train_y)
    r3 = pearsonr(train_x[:,2], train_y)

    return r1.statistic, r2.statistic, r3.statistic

def calculate_r2(train_ds, test_ds):
    train_x = train_ds.get_x()
    train_y = train_ds.get_y()
    test_x = test_ds.get_x()
    test_y = test_ds.get_y()

    model_instance = LinearRegression()
    model_instance = model_instance.fit(train_x[:,0:1], train_y)
    r1 = model_instance.score(train_x[:,0:1], train_y)

    model_instance = LinearRegression()
    model_instance = model_instance.fit(train_x[:,1:2], train_y)
    r2 = model_instance.score(train_x[:,1:2], train_y)

    model_instance = LinearRegression()
    model_instance = model_instance.fit(train_x[:,2:3], train_y)
    r3 = model_instance.score(train_x[:,2:3], train_y)

    return r1, r2, r3


def r(dm):
    i0s = []
    i1s = []
    i2s = []
    for train_ds, test_ds in dm.get_k_folds():
        i0, i1, i2 = calculate_r(train_ds, test_ds)
        i0s.append(i0)
        i1s.append(i1)
        i2s.append(i2)

    return i0s, i1s, i2s

def r2(dm):
    i0s = []
    i1s = []
    i2s = []
    for train_ds, test_ds in dm.get_k_folds():
        i0, i1, i2 = calculate_r2(train_ds, test_ds)
        i0s.append(i0)
        i1s.append(i1)
        i2s.append(i2)

    return i0s, i1s, i2s


for cs in ["rgb", "hsv", "cielab"]:
    r1_1s = []
    r1_2s = []
    r1_3s = []

    r2_1s = []
    r2_2s = []
    r2_3s = []

    for ds in ["lucas", "raca", "ossl"]:
        dm = ds_manager.DSManager(ds, cs)
        i0s, i1s, i2s = r(dm)
        print(f"{cs} {ds}")
        print(f"r")

        r1_1 = np.round(sum(i0s) / len(i0s),3)
        print(r1_1)
        r1_2 = np.round(sum(i1s) / len(i1s),3)
        print(r1_2)
        r1_3 = np.round(sum(i2s) / len(i2s),3)
        print(r1_3)

        r1_1s.append(r1_1)
        r1_2s.append(r1_2)
        r1_3s.append(r1_3)

        i0s, i1s, i2s = r2(dm)
        print(f"r2")

        r2_1 = np.round(sum(i0s) / len(i0s),3)
        print(r2_1)
        r2_2 = np.round(sum(i1s) / len(i1s),3)
        print(r2_2)
        r2_3 = np.round(sum(i2s) / len(i2s),3)
        print(r2_3)

        r2_1s.append(r2_1)
        r2_2s.append(r2_2)
        r2_3s.append(r2_3)

    print(cs)
    print("=======r1-AVG=====")

    r1_1avg = np.round(sum(r1_1s) / len(r1_1s), 3)
    print(r1_1avg)
    r1_2avg = np.round(sum(r1_2s) / len(r1_2s), 3)
    print(r1_2avg)
    r1_3avg = np.round(sum(r1_3s) / len(r1_3s), 3)
    print(r1_3avg)

    print("=======r2-AVG=====")

    r2_1avg = np.round(sum(r2_1s) / len(r2_1s), 3)
    print(r2_1avg)
    r2_2avg = np.round(sum(r2_2s) / len(r2_2s), 3)
    print(r2_2avg)
    r2_3avg = np.round(sum(r2_3s) / len(r2_3s), 3)
    print(r2_3avg)
