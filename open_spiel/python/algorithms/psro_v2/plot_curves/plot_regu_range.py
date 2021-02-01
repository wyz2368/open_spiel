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

do_mean = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')[:103]
do_std = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')[:103]

crd_035_mean = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_mean.csv', delimiter=',')
crd_035_std = genfromtxt('./data/2Nash_merged_csv/dqn_crd_0.35_Deepmind_std.csv', delimiter=',')

crd_008_mean = genfromtxt('./regularization_data/crd_008_mean.csv', delimiter=',')
crd_008_std = genfromtxt('./regularization_data/crd_008_std.csv', delimiter=',')

crd_010_mean = genfromtxt('./regularization_data/crd_010_mean.csv', delimiter=',')
crd_010_std = genfromtxt('./regularization_data/crd_010_std.csv', delimiter=',')

crd_015_mean = genfromtxt('./regularization_data/crd_015_mean.csv', delimiter=',')
crd_015_std = genfromtxt('./regularization_data/crd_015_std.csv', delimiter=',')

crd_020_mean = genfromtxt('./regularization_data/crd_020_mean.csv', delimiter=',')
crd_020_std = genfromtxt('./regularization_data/crd_020_std.csv', delimiter=',')

crd_030_mean = genfromtxt('./regularization_data/crd_030_mean.csv', delimiter=',')
crd_030_std = genfromtxt('./regularization_data/crd_030_std.csv', delimiter=',')

crd_040_mean = genfromtxt('./regularization_data/crd_040_mean.csv', delimiter=',')
crd_040_std = genfromtxt('./regularization_data/crd_040_std.csv', delimiter=',')

crd_050_mean = genfromtxt('./regularization_data/crd_050_mean.csv', delimiter=',')
crd_050_std = genfromtxt('./regularization_data/crd_050_std.csv', delimiter=',')


axes = plt.gca()
axes.set_ylim([0.5,2])

X = np.arange(1, 104)

# plt.plot(X, Mike_fic_mean, color="C2", label='FP')

plt.plot(X, do_mean, color="C1", marker="1", label='NE-based regret of DO')
plt.fill_between(X, do_mean+do_std, do_mean-do_std, alpha=0.1, color="C1")

plt.plot(X, do_crd_mean, color="b", marker="*", label='RRD-based regret of DO')
plt.fill_between(X, do_crd_mean+do_crd_std, do_crd_mean-do_crd_std, alpha=0.1, color="C1")

plt.plot(X, crd_008_mean, color="C2", label='RRD, lam=0.08')
plt.fill_between(X, crd_008_mean+crd_008_std, crd_008_mean-crd_008_std, alpha=0.1, color="C2")

plt.plot(X, crd_010_mean, color="C3", label='RRD, lam=0.10')
plt.fill_between(X, crd_010_mean+crd_010_std, crd_010_mean-crd_010_std, alpha=0.1, color="C3")

plt.plot(X, crd_015_mean, color="C4", label='RRD, lam=0.15')
plt.fill_between(X, crd_015_mean+crd_015_std, crd_015_mean-crd_015_std, alpha=0.1, color="C4")

plt.plot(X, crd_020_mean, color="C5", label='RRD, lam=0.20')
plt.fill_between(X, crd_020_mean+crd_020_std, crd_020_mean-crd_020_std, alpha=0.1, color="C5")

plt.plot(X, crd_030_mean, color="C6", label='RRD, lam=0.30')
plt.fill_between(X, crd_030_mean+crd_030_std, crd_030_mean-crd_030_std, alpha=0.1, color="C6")

plt.plot(X, crd_035_mean, color="C7", label='RRD, lam=0.35')
plt.fill_between(X, crd_035_mean+crd_035_std, crd_035_mean-crd_035_std, alpha=0.1, color="C7")

plt.plot(X, crd_040_mean, color="C8", label='RRD, lam=0.40')
plt.fill_between(X, crd_040_mean+crd_040_std, crd_040_mean-crd_040_std, alpha=0.1, color="C8")

plt.plot(X, crd_050_mean, color="C9", label='RRD, lam=0.50')
plt.fill_between(X, crd_050_mean+crd_050_std, crd_050_mean-crd_050_std, alpha=0.1, color="C9")

plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 12})

plt.show()

