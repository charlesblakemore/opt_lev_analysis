import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as signal
import scipy.optimize as opti
from scipy.optimize import curve_fit
import re, sys

plt.rcParams.update({'font.size': 14})


paths = [#'/daq2/20190618/pramp_tests/outgassing', \
         #'/daq2/20190624/pramp/outgassing_test1', \
         #'/daq2/20190625/pramp/outgassing_test2', \
         '/daq2/20190626/bead1/spinning/pramp/outgassing/50kHz_4Vpp', \
         '/daq2/20190626/bead1/spinning/pramp/outgassing/50kHz_4Vpp_2', \
         '/daq2/20190626/bead1/spinning/pramp/outgassing/50kHz_4Vpp_after-SF6', \
         '/daq2/20190626/bead1/spinning/pramp/outgassing/after_He']


save_dir = '/processed_data/spinning/pramp_data/20190626/outgassing/'


def line(x, a, b):
    return a * x + b


for pathind, path in enumerate(paths):
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
    abs_time = times[0]
    times = np.array(times) - abs_time

    bara_avg = 0.5 * (pressures[:,1] + pressures[:,2]) * 1.3

    plt.figure(1)
    plt.ion()

    plt.plot(times, bara_avg)

    plt.xlabel('Time [s]')
    plt.ylabel('Chamber Pressure [mbar]')
    plt.tight_layout()
    plt.show()

    try:
        guess = input('Fit times (lower, upper): ')
        lower, upper = list(map(float, guess.split(',')))

        fit_inds = (times > lower) * (times < upper)

        popt, pcov = opti.curve_fit(line, times[fit_inds], bara_avg[fit_inds])

        plt.plot(times[fit_inds], line(times[fit_inds], *popt), '--', lw=2, color='r')
        plt.draw()

        print('Plotting linear fit for 5 seconds...')
        print('    found {:0.4g} mbar/s'.format(popt[0]))
        plt.pause(5)

        plt.close(1)
        plt.ioff()

    except:
        plt.close(1)
        plt.ioff()
        continue


    filename = save_dir + 'outgassing-at-time_{:d}.txt'.format(int(abs_time*1e9))

    file = open(filename, 'wb')
    file.write('# outgassing rate in mbar/s\n')
    file.write('{:d}\n'.format(np.int(abs_time*1e9)))
    file.write('{:g}\n'.format(popt[0]))
    file.close()
