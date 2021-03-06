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


Mike_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_mean.csv', delimiter=',')
Mike_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_std.csv', delimiter=',')

deepmind_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_fic_deepmind_mean.csv', delimiter=',')
deepmind_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_fic_deepmind_std.csv', delimiter=',')

# Mike_crd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_mean.csv', delimiter=',')
# Mike_crd_std = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_std.csv', delimiter=',')

dqn_do_mean = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')
dqn_do_std = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')

dqn_do_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_do_fic_mean.csv', delimiter=',')
dqn_do_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_do_fic_std.csv', delimiter=',')

# Mike_prd1e5_mean = genfromtxt('./regularization_data/dqn_prd_1e5_deepmind_mean.csv', delimiter=',')
# Mike_prd1e5_std = genfromtxt('./regularization_data/dqn_prd_1e5_deepmind_std.csv', delimiter=',')
#
# Mike_prd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_prd_deepmind_mean.csv', delimiter=',')[:103]
# Mike_prd_std = genfromtxt('./data/2Nash_merged_csv/dqn_prd_deepmind_std.csv', delimiter=',')[:103]

axes = plt.gca()
axes.set_ylim([0.5,2.5])

X = np.arange(1, 151)

# plt.plot(X, Mike_fic_mean, color="C2", label='FP')



plt.plot(X, deepmind_fic_mean, color="C2", label='uniform-based regret of FP')
plt.fill_between(X, deepmind_fic_mean+deepmind_fic_std, deepmind_fic_mean-deepmind_fic_std, alpha=0.1, color="C2")

plt.plot(X, dqn_do_fic_mean, color="C0", label='uniform-based regret of DO')
plt.fill_between(X, dqn_do_fic_mean+dqn_do_fic_std, dqn_do_fic_mean-dqn_do_fic_std, alpha=0.1, color="C0")

plt.plot(X, Mike_fic_mean, color="C5", label='NE-based regret of FP')
plt.fill_between(X, Mike_fic_mean+Mike_fic_std, Mike_fic_mean-Mike_fic_std, alpha=0.1, color="C5")

plt.plot(X, dqn_do_mean, color="C1", label='NE-based regret of DO')
plt.fill_between(X, dqn_do_mean+dqn_do_std, dqn_do_mean-dqn_do_std, alpha=0.1, color="C1")

# plt.plot(X, Mike_prd1e5_mean, color="C7", label='PRD')
# plt.fill_between(X, Mike_prd1e5_mean+Mike_prd1e5_std, Mike_prd1e5_mean-Mike_prd1e5_std, alpha=0.1, color="C7")
#
# plt.plot(X, Mike_crd_mean, color="C2", label='RRD with regret shreshold')
# plt.fill_between(X, Mike_crd_mean+Mike_crd_std, Mike_crd_mean-Mike_crd_std, alpha=0.1, color="C2")
#
# plt.plot(X, Mike_prd_mean, color="C0", label='RRD with fixed number of iterations')
# plt.fill_between(X, Mike_prd_mean+Mike_prd_std, Mike_prd_mean-Mike_prd_std, alpha=0.1, color="C0")




plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 16})

plt.show()

