import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss


data = np.loadtxt("/home/arider/opt_lev_analysis/scripts/phase_drift_reconstruction/data/x_y_sum_ref_oscscope_20160314.csv", skiprows = 2, delimiter = ',')

def mean_sub(arr):
    #mean subtracts an 1d array.
    return arr - np.mean(arr)

t = data[:, 0]
x = mean_sub(data[:, 1])
y = mean_sub(data[:, 2])
s = mean_sub(data[:, 3])
refc = mean_sub(data[:, 4])
refs = np.fft.irfft(np.fft.rfft(refc)*np.exp(0.5j*np.pi))


plt.plot(t, refc)
plt.plot(t, refs)
plt.show()
