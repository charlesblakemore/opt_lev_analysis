import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re, sys

# Define directory to use, save location, data/drive axes and
# the frequency parameters: expected rotation frequency and the
# bandwith for bandpass filtering
save = True

path = '/daq2/20190514/bead1/spinning/pramp3/He/50kHz_4Vpp_4'

drive_ax = 1
data_ax = 0

out_f = "/processed_data/spinning/pramp_data/20190514/He/50kHz_4Vpp_3_4"

f_rot = 50000   # [Hz]
bw = 1.         # [Hz]

plot_dat = False

################################################################
################################################################

# Make sure save path is valid. Creates directories with read/write
# permissions for only the current user
bu.make_all_pardirs(out_f)

# Find all the relevant datafiles, sort them by time and 
# subselect some files if desired. Default is no subselection
files, lengths = bu.find_all_fnames(path, sort_time=True)
init_file = 0
final_file = len(files)
n_file = final_file-init_file

# Pull out attributes common to all files
obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']

# Construct time and frequency arrays from sampling parameters
tarr0 = np.linspace(0, (Ns-1)/Fs, Ns)
freqs = np.fft.rfftfreq(Ns, d = 1./Fs)

# Define array to be filled as we process data
times = np.zeros(n_file)
phases = np.zeros(n_file)
dphases = np.zeros(n_file)
pressures = np.zeros((n_file, 3))

# Construct simple top-hat filters to pick out only the rotation
# peak of interest: at frot for the drive and 2*frot for the 
# polarization rotation signal
f_rot2 = 2.*f_rot
finds = np.abs(freqs - f_rot) > bw/2.
finds2 = np.abs(freqs - f_rot2) > bw/2.

# Loop over files
for i, f in enumerate(files[init_file:final_file]):
    bu.progress_bar(i, n_file)
    sys.stdout.flush()
    # Analysis nested in try/except block just in case there
    # is a corrupted file or something
    try:
        # Load data, computer FFTs and plot if requested
        obj = hsDat(f)
        dat_fft = np.fft.rfft(obj.dat[:, data_ax])
        drive_fft = np.fft.rfft(obj.dat[:, drive_ax])
        if plot_dat:
            plt.figure()
            plt.loglog(freqs, np.abs(dat_fft))
            plt.loglog(freqs, np.abs(drive_fft))

        # Filter data outside the window of interest and plot
        # if requested
        dat_fft[finds2] = 0.
        drive_fft[finds] = 0.
        if plot_dat:
            plt.loglog(freqs, np.abs(dat_fft))
            plt.loglog(freqs, np.abs(drive_fft))
            plt.show()

        # Compute the raw phases of drive and response
        #    drive: phase of frot signal
        #    response: phase of 2*frot signal
        phases[i] = np.angle(np.sum(dat_fft))
        dphases[i] = np.angle(np.sum(drive_fft))
        pressures[i, :] = obj.attribs["pressures"]
        times[i] = obj.attribs['time']
    except:
        print "bad file"

# There is likely an artificial, unknown phase offset (which we
# assume to be constant) introduced by various front-end electronics
# and the digitizer itself. This phase offset can be corrected for 
# later. Because this constant exists, you can invert the drive phase
# and not lose information about the phase difference. We choose to
# have all drive phases be positive 
dphases[dphases<0.] += np.pi

# Since sin(theta)^2 = 0.5*(1 - cos(2*theta)), there is a factor
# of two between the phase of drive and response.
delta_phi = phases - 2.*dphases

# Put all the phase between pi and -pi for better unwrapping 
# later when we do more analysis
delta_phi[delta_phi > np.pi] -= 2.*np.pi
delta_phi[delta_phi < -np.pi] += 2.*np.pi

if save:
    np.save(out_f + '_phi.npy', 0.5 * delta_phi)
    np.save(out_f + "_pressures.npy", pressures)
    np.save(out_f + "_time.npy", times)
