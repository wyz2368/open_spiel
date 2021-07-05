from numpy import genfromtxt
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl
import math

from scipy.signal import savgol_filter
# yhat = savgol_filter(y, 51, 2) # window size 51, polynomial order 3

window_size = 55
order = 2

########### Plot alpha rank filter ############

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.title("NashConv Curves ", fontsize = 22)

regret_mean = genfromtxt('./regularization_data/regret_fixed_RD_iter_mean.csv', delimiter=',')[:30]
regret_std = genfromtxt('./regularization_data/regret_fixed_RD_iter_std.csv', delimiter=',')

X = np.arange(1, 31)

plt.plot(X, regret_mean, color="C1")

plt.xticks(size = 18)
plt.yticks(size = 18)

plt.xlabel('Number of Iterations', fontsize = 23)
plt.ylabel('Regret', fontsize = 20)

# plt.legend(loc="best", prop={'size': 20})

plt.show()