import os
os.chdir("../")

import ds_manager

d = ds_manager.DSManager("lucas", "rgb", si=["soci", "ibs"], normalize=False)
ds = d.test_ds
data, soc = ds[0]
soci = data[2] / (data[0] * data[1])
ibs = 1/(data[2]**2)
print(data)
print(soci)
print(ibs)


