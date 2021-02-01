from numpy import genfromtxt
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

plt.figure()


do_crd_mean = genfromtxt('./regularization_data/dqn_do_crd_mean.csv', delimiter=',')[:103]
do_crd_std = genfromtxt('./regularization_data/dqn_do_crd_std.csv', delimiter=',')[:103]

crd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_mean.csv', delimiter=',')
crd_std = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_std.csv', delimiter=',')

do_mean = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')[:103]
do_std = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')[:103]

axes = plt.gca()
axes.set_ylim([0.5,2])

X = np.arange(1, 104)

# plt.plot(X, Mike_fic_mean, color="C2", label='FP')

plt.plot(X, do_mean, color="C1", label='NE-based regret of DO')
plt.fill_between(X, do_mean+do_std, do_mean-do_std, alpha=0.1, color="C1")

plt.plot(X, do_crd_mean, color="C0", label='RRD-based regret of DO')
plt.fill_between(X, do_crd_mean+do_crd_std, do_crd_mean-do_crd_std, alpha=0.1, color="C1")

plt.plot(X, crd_mean, color="C2", label='RRD-based regret of RRD')
plt.fill_between(X, crd_mean+crd_std, crd_mean-crd_std, alpha=0.1, color="C2")

plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 16})

plt.show()

