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

# dqn_abs = savgol_filter(genfromtxt('./data/dqn_abs.csv', delimiter=','), window_size, order)
# dqn_do = savgol_filter(genfromtxt('./data/dqn_do.csv', delimiter=','), window_size, order)
# dqn_fic = savgol_filter(genfromtxt('./data/dqn_fic.csv', delimiter=','), window_size, order)
# dqn_prd = savgol_filter(genfromtxt('./data/dqn_prd.csv', delimiter=','), window_size, order)
# dqn_do_uniform = savgol_filter(genfromtxt('./data/dqn_do_uniform.csv', delimiter=','), window_size, order)


# dqn_abs = genfromtxt('./data/dqn_abs.csv', delimiter=',')
# dqn_do = genfromtxt('./data/dqn_do.csv', delimiter=',')
# dqn_fic = genfromtxt('./data/dqn_fic.csv', delimiter=',')
# dqn_prd = genfromtxt('./data/dqn_prd.csv', delimiter=',')
# dqn_do_uniform = genfromtxt('./data/dqn_do_uniform.csv', delimiter=',')

# axes = plt.gca()
# axes.set_ylim([0.5,4])

# x = np.arange(1, len(dqn_abs)+1)
# plt.plot(x, dqn_abs, '-C1', label= "HBS")
# plt.plot(x, dqn_do, '-C5', label= "DO")
# plt.plot(x, dqn_fic, '-C4', label= "Uniform")
# plt.plot(x, dqn_prd, '-C3', label= "PRD")
# plt.plot(x, dqn_do_uniform, '-C2', label= "DO+Unifrom")
#
#
# plt.xlabel("Number of Iterations")
# plt.ylabel("NashConv")
# plt.title("Average NashConv over 10 runs in Leduc Poker")
# plt.legend(loc="best")
# plt.show()



################### Draw different NashConvs  ##########################
# deepmind_fic = savgol_filter(genfromtxt('./data/2Nash_merged_csv/deepmind_fic.csv', delimiter=','), window_size, order)
# Mike_fic = savgol_filter(genfromtxt('./data/2Nash_merged_csv/Mike_fic.csv', delimiter=','), window_size, order)
# dqn_do = savgol_filter(genfromtxt('./data/2Nash_merged_csv/dqn_do.csv', delimiter=','), window_size, order)

deepmind_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_fic_deepmind_mean.csv', delimiter=',')
Mike_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_mean.csv', delimiter=',')
dqn_do_mean = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')
deepmind_prd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_prd_deepmind_mean.csv', delimiter=',')
Mike_prd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_prd_Mike_mean.csv', delimiter=',')

deepmind_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_fic_deepmind_std.csv', delimiter=',')
Mike_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_std.csv', delimiter=',')
dqn_do_std = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')
deepmind_prd_std = genfromtxt('./data/2Nash_merged_csv/dqn_prd_deepmind_std.csv', delimiter=',')
Mike_prd_std = genfromtxt('./data/2Nash_merged_csv/dqn_prd_Mike_std.csv', delimiter=',')

dqn_do_prd_prd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_do_prd_prd_mean.csv', delimiter=',')
dqn_do_prd_prd_std = genfromtxt('./data/2Nash_merged_csv/dqn_do_prd_prd_std.csv', delimiter=',')

axes = plt.gca()
axes.set_ylim([0.5,2])

x = np.arange(1, 151)
plt.plot(x, dqn_do_mean, '-b', label= "NE-based regret of DO")
plt.fill_between(x, dqn_do_mean+dqn_do_std, dqn_do_mean-dqn_do_std, alpha=0.1, color="b")

plt.plot(x, deepmind_fic_mean, '-C2', label= "uniform-based regret of FP")
plt.fill_between(x, deepmind_fic_mean+deepmind_fic_std, deepmind_fic_mean-deepmind_fic_std, alpha=0.1, color="C2")
#
plt.plot(x, Mike_fic_mean, '-C1', label= "NE-based regret of FP")
plt.fill_between(x, Mike_fic_mean+Mike_fic_std, Mike_fic_mean-Mike_fic_std, alpha=0.1, color="C1")

# plt.plot(x, deepmind_prd_mean, '-C2', label= "PRD-based regret of PRD")
# plt.fill_between(x, deepmind_prd_mean+deepmind_prd_std, deepmind_prd_mean-deepmind_prd_std, alpha=0.1, color="C2")
#
# plt.plot(x, Mike_prd_mean, '-C1', label= "NE-based regret of PRD")
# plt.fill_between(x, Mike_prd_mean+Mike_prd_std, Mike_prd_mean-Mike_prd_std, alpha=0.1, color="C1")

# plt.plot(x, dqn_do_prd_prd_mean, '-C5', label= "PRD-based regret of DO")
# plt.fill_between(x, dqn_do_prd_prd_mean+dqn_do_prd_prd_std, dqn_do_prd_prd_mean-dqn_do_prd_prd_std, alpha=0.1, color="C5")

plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)


# plt.xlabel("Number of Iterations")
# plt.ylabel("NashConv")
# plt.title("NashConvs under Different Metrics")
plt.legend(loc="best", prop={'size': 16})
plt.show()