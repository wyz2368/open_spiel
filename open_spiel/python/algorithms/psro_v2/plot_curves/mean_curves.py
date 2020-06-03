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
deepmind_fic = savgol_filter(genfromtxt('./data/deepmind_fic.csv', delimiter=','), window_size, order)
Mike_fic = savgol_filter(genfromtxt('./data/Mike_fic.csv', delimiter=','), window_size, order)
dqn_do = savgol_filter(genfromtxt('./data/dqn_do.csv', delimiter=','), window_size, order)

# deepmind_fic = genfromtxt('./data/deepmind_fic.csv', delimiter=',')
# Mike_fic = genfromtxt('./data/Mike_fic.csv', delimiter=',')
# dqn_do = genfromtxt('./data/dqn_do.csv', delimiter=',')

x = np.arange(1, len(dqn_do)+1)
plt.plot(x, dqn_do, '-C1', label= "DO")
plt.plot(x, deepmind_fic, '-b', label= "Heuristic-based (uniform)")
plt.plot(x, Mike_fic, '-C2', label= "NE-based (uniform)")


plt.xlabel("Number of Iterations")
plt.ylabel("NashConv")
plt.title("NashConvs under Different Metrics")
plt.legend(loc="best")
plt.show()