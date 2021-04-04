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

crd_035_mean_Mike = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')[:103]
crd_035_std_Mike = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')[:103]

crd_035_mean_deepmind = genfromtxt('./regularization_data/dqn_do_crd_mean.csv', delimiter=',')[:103]
crd_035_std_deepmind = genfromtxt('./regularization_data/dqn_do_crd_std.csv', delimiter=',')[:103]

axes = plt.gca()
axes.set_ylim([0.5, 2])

X = np.arange(1, 104)

# plt.plot(X, Mike_fic_mean, color="C2", label='FP')

plt.plot(X, crd_035_mean_Mike, color="C1", label='Regret of NE')
plt.fill_between(X, crd_035_mean_Mike+crd_035_std_Mike, crd_035_mean_Mike-crd_035_std_Mike, alpha=0.1, color="C1")

plt.plot(X, crd_035_mean_deepmind, color="C2", label='Regret of RRD target')
plt.fill_between(X, crd_035_mean_deepmind+crd_035_std_deepmind, crd_035_mean_deepmind-crd_035_std_deepmind, alpha=0.1, color="C2")




plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 16})

plt.show()
