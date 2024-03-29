import spectral_dataset
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score

ds = spectral_dataset.SpectralDataset(is_train=True)
x = ds.get_x()
y = ds.get_y()
reg = LinearRegression().fit(x,y)

print("Train done")

pickle.dump(reg, open("../models/linear", "wb"))

ds = spectral_dataset.SpectralDataset(is_train=False)
x = ds.get_x()
y = ds.get_y()

y_hat = reg.predict(x)
print(r2_score(y, y_hat))

