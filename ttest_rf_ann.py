from scipy import stats
import numpy as np

rf = np.array([
0.707,
0.784,
0.638,
0.752,
0.725,
0.698,
0.767,
0.743,
0.676,
0.784,
0.428,
0.443,
0.435,
0.429,
0.432,
0.413,
0.441,
0.441,
0.456,
0.432,
0.724,
0.717,
0.682,
0.71,
0.698,
0.75,
0.738,
0.742,
0.72,
0.7
])

ann = np.array([
0.711,
0.785,
0.634,
0.76,
0.743,
0.698,
0.77,
0.735,
0.676,
0.776,
0.435,
0.441,
0.441,
0.443,
0.435,
0.414,
0.458,
0.451,
0.472,
0.435,
0.744,
0.731,
0.691,
0.732,
0.714,
0.751,
0.734,
0.762,
0.74,
0.717,
])
print(np.mean(ann), np.mean(rf))
x = stats.ttest_rel(ann, rf)
print(x)