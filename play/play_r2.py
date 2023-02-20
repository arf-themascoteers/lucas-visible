from sklearn.linear_model import LinearRegression
import os
os.chdir("../")
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

npdf = pd.read_csv("data/lucas/hue.csv").to_numpy()
X = npdf[:,0:1]
y = npdf[:,-1]

print(r2_score(X,y))

model = LinearRegression().fit(X,y)
print(model.score(X,y))

y_hat = model.predict(X)


print(r2_score(y, y_hat))
y_hat = y_hat.reshape(-1,1)
model = LinearRegression().fit(y_hat,y)
print(model.score(y_hat,y))
