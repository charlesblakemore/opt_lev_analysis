import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu




save_dir = '/processed_data/spinning/pramp_data/20190626/outgassing/'

files, lengths = bu.find_all_fnames(save_dir, ext='.txt')


times = []
rates = []
for filename in files:
    file_obj = open(filename, 'rb')
    lines = file_obj.readlines()
    file_obj.close()

    time = float(int(lines[1])) * 1.0e-9
    rate = float(lines[2])

    times.append(time)
    rates.append(rate)

times = np.array(times)
rates = np.array(rates)

sort_inds = np.argsort(times)

times = times[sort_inds]
rates = rates[sort_inds]

plt.plot((times-times[0]) / 3600., rates)
plt.show()