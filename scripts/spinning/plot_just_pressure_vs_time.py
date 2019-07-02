import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re, sys

plt.rcParams.update({'font.size': 14})


paths = ['/daq2/20190618/pramp_tests/outgassing', \
		 #'/daq2/20190624/pramp/outgassing_test1', \
		 '/daq2/20190625/pramp/outgassing_test2', \
		 '/daq2/20190626/bead1/spinning/pramp/outgassing/50kHz_4Vpp']

labels = ['Old data', 'Reseated Window', 'Much Later']


chan_to_plot = 2

time_arrs = []
pressure_arrs = []

for path in paths:
	files, lengths = bu.find_all_fnames(path, sort_time=True)
	nfiles = len(files)
	init_file = 0

	times = []
	pressures = []

	for fileind, file in enumerate(files):
		bu.progress_bar(fileind, nfiles)
		obj = hsDat(file)
		pressures.append(obj.attribs["pressures"])
		times.append(obj.attribs["time"] * 1e-9)

	pressures = np.array(pressures)
	times = np.array(times) - times[0]

	pressure_arrs.append(pressures)
	time_arrs.append(times)

for pathind, path in enumerate(paths):
	plt.plot(time_arrs[pathind], pressure_arrs[pathind][:,chan_to_plot], \
			 label=labels[pathind])
plt.xlabel('Time [s]')
plt.ylabel('Chamber Pressure [torr]')
plt.suptitle('Leak Got Better')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(top=0.91)
plt.show()
