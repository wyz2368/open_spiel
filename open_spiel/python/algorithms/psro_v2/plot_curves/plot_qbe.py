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


Mike_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_mean.csv', delimiter=',')[:103]
Mike_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_std.csv', delimiter=',')[:103]

Mike_crd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_mean.csv', delimiter=',')
Mike_crd_std = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_std.csv', delimiter=',')

dqn_do_mean = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')[:103]
dqn_do_std = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')[:103]

qbe_mean = genfromtxt('./regularization_data/qbe_025_mean.csv', delimiter=',')
qbe_std = genfromtxt('./regularization_data/qbe_025_std.csv', delimiter=',')


axes = plt.gca()
axes.set_ylim([0.5,2])

X = np.arange(1, 104)

# plt.plot(X, Mike_fic_mean, color="C2", label='FP')

plt.plot(X, Mike_fic_mean, color="C5", label='FP')
plt.fill_between(X, Mike_fic_mean+Mike_fic_std, Mike_fic_mean-Mike_fic_std, alpha=0.1, color="C5")

plt.plot(X, dqn_do_mean, color="C1", label='DO')
plt.fill_between(X, dqn_do_mean+dqn_do_std, dqn_do_mean-dqn_do_std, alpha=0.1, color="C1")

plt.plot(X, qbe_mean, color="C7", label='QBE with regret shreshold')
plt.fill_between(X, qbe_mean+qbe_std, qbe_mean-qbe_std, alpha=0.1, color="C7")

plt.plot(X, Mike_crd_mean, color="C2", label='RRD with regret shreshold')
plt.fill_between(X, Mike_crd_mean+Mike_crd_std, Mike_crd_mean-Mike_crd_std, alpha=0.1, color="C2")





plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 16})

plt.show()
