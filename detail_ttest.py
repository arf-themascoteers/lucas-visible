import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("play/d12.csv")
results = df.to_numpy()

pvalues = np.zeros((results.shape[1], results.shape[1]))

for i in range(results.shape[1]):
    for j in range(results.shape[1]):
        alg_perf_1 = results[:,i]
        alg_perf_2 = results[:,j]
        x = stats.ttest_rel(alg_perf_1, alg_perf_2)
        pvalues[i,j] = x.pvalue
        if math.isnan(pvalues[i,j]):
            pvalues[i, j] = 0

#columns = list(df2.columns)
#columns = ["RF-HSV","RF-CIE L*a*b*","ANN-HSV","ANN-CIE L*a*b*"]
columns = ["RF-RGB","RF-HSV","RF-L*a*b*","ANN-RGB","ANN-HSV","ANN-L*a*b*"]

mask = np.triu(np.ones_like(pvalues, dtype=bool))
for i in range(mask.shape[0]):
    mask[i][i] = True
sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(pvalues, mask = mask, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5,"location":"top"},
            xticklabels=columns, yticklabels=columns, annot=True, annot_kws={'size':10}
            )
ax.invert_yaxis()
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.subplots_adjust(bottom=0.4)
plt.show()