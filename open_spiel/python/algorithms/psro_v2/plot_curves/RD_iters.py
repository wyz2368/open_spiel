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

RD_iter = genfromtxt('./regularization_data/RD_iter1.csv', delimiter=',')
# RD_converge = np.ones(len(RD_iter)) * 1e5

# axes = plt.gca()
# axes.set_ylim([0,3])

X = np.arange(1, len(RD_iter)+1)


# plt.plot(X, RD_converge, color="C1", label='Convergence to NE')
plt.plot(X, RD_iter, color="C2", label='Regularized RD')


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations of PSRO', fontsize = 22)
plt.ylabel('Number of RD Stopping Iterations', fontsize = 19)

plt.legend(loc="best", prop={'size': 17})

plt.show()

