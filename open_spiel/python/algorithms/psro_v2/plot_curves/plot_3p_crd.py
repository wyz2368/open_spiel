from numpy import genfromtxt
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from scipy.signal import savgol_filter
# yhat = savgol_filter(y, 51, 2) # window size 51, polynomial order 3

window_size = 55
order = 2

########### Plot alpha rank filter ############

plt.figure()
# plt.title("NashConv Curves ", fontsize = 22)

crd_005_mean = genfromtxt('./regularization_data/RD_crd_005_mean.csv', delimiter=',')
crd_005_std = genfromtxt('./regularization_data/RD_crd_005_std.csv', delimiter=',')

# crd_030_mean = genfromtxt('./regularization_data/RD_crd_030_mean.csv', delimiter=',')
# crd_030_std = genfromtxt('./regularization_data/RD_crd_030_std.csv', delimiter=',')

crd_060_mean = genfromtxt('./regularization_data/RD_crd_060_mean.csv', delimiter=',')
crd_060_std = genfromtxt('./regularization_data/RD_crd_060_std.csv', delimiter=',')


axes = plt.gca()
axes.set_ylim([2,8])

X = np.arange(1, 24)

# plt.plot(X, Mike_fic_mean, color="C2", label='FP')

plt.plot(X, crd_005_mean, color="C3", label='RRD-based regret of Nash')
plt.fill_between(X, crd_005_mean+crd_005_std, crd_005_mean-crd_005_std, alpha=0.1, color="C3")

# plt.plot(X, crd_030_mean, color="C1", label='RRD-based regret of RRD (lam=0.3)')
# plt.fill_between(X, crd_030_mean+crd_030_std, crd_030_mean-crd_030_std, alpha=0.1, color="C1")

plt.plot(X, crd_060_mean, color="C2", label='RRD-based regret of RRD (lam=0.6)')
plt.fill_between(X, crd_060_mean+crd_060_std, crd_060_mean-crd_060_std, alpha=0.1, color="C2")




plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 19})

plt.show()
