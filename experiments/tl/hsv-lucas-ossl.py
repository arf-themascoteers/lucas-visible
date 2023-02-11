import os
import pandas as pd
import numpy as np
from scipy import stats
os.chdir("../../")
import torch
from train import train
from test import test
import ds_manager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(lucas_train, lucas_test, ossl_train, ossl_test):
    lucas_test, ossl_test = ds_manager.DSManager.equalize_datasets(lucas_test, ossl_test)
    ossl_train = ds_manager.DSManager.minify_datasets(ossl_train, 0.1)
    lucas_model = train(device, lucas_train)
    lucas_result = test(device, lucas_test, lucas_model)
    ossl_result_lucas = test(device, ossl_test, lucas_model)
    calibrated_model = train(device, ossl_train, model=lucas_model)
    ossl_result_calibrated = test(device, ossl_test, calibrated_model)
    ossl_model = train(device, ossl_train)
    ossl_result_small_train = test(device, ossl_test, ossl_model)
    return lucas_result, ossl_result_lucas, ossl_result_calibrated, ossl_result_small_train


lucas = ds_manager.DSManager("lucas","hsv")
ossl = ds_manager.DSManager("ossl","hsv")

results = np.zeros((10,4))

for lucas_index, (lucas_train, lucas_test) in enumerate(lucas.get_k_folds()):
    for ossl_index, (ossl_train, ossl_test) in enumerate(ossl.get_k_folds()):
        if lucas_index == ossl_index:
            lucas_result, ossl_result_lucas, ossl_result_calibrated, ossl_result_small_train = run_experiment(lucas_train, lucas_test, ossl_train, ossl_test)
            results[lucas_index][0] = lucas_result
            results[lucas_index][1] = ossl_result_lucas
            results[lucas_index][2] = ossl_result_calibrated
            results[lucas_index][3] = ossl_result_small_train
            print(lucas_result, ossl_result_lucas, ossl_result_calibrated, ossl_result_small_train)

df = pd.DataFrame(results, columns=['lucas', 'no-calibration', 'calibrated', 'ossl-trained'])
df.to_csv("tl.csv", index=False)
print("done")

cali_result = results[:,2]
ossl_result = results[:,3]
print(np.mean(cali_result), np.mean(ossl_result))
x = stats.ttest_rel(cali_result, ossl_result)
print(x)






