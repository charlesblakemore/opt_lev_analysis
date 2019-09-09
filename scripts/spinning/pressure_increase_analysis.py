import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as signal
import scipy.optimize as opti
import re, sys

import dill as pickle

import bead_util as bu

import itertools

# Define directory to use, save location, data/drive axes and
# the frequency parameters: expected rotation frequency and the
# bandwith for bandpass filtering
save = True

#gas = 'He'
#pramp_ind = 3

tabor_mon_fac = 100.

def sine(x, A, f, phi, c):
    return A * np.sin(2*np.pi*f*x + phi) + c

gases = ['He', 'N2', 'Ar', 'Kr', 'Xe', 'SF6']

inds = [1,2,3]

for meas in itertools.product(gases, inds):
    gas, pramp_ind = meas
    path = '/daq2/20190626/bead1/spinning/pramp/{:s}/50kHz_4Vpp_{:d}'.format(gas, pramp_ind)

    drive_ax = 1
    data_ax = 0

    out_f = '/processed_data/spinning/pramp_data/20190626/{:s}/50kHz_4Vpp_{:d}'.format(gas, pramp_ind)

    f_rot = 50000   # [Hz]
    bw_filt = 2000
    bw = 1.0        # [Hz]
    plot_dat = False

    ################################################################
    ################################################################

    # Make sure save path is valid. Creates directories with read/write
    # permissions for only the current user
    bu.make_all_pardirs(out_f)

    # Find all the relevant datafiles, sort them by time and 
    # subselect some files if desired. Default is no subselection
    files, lengths = bu.find_all_fnames(path, sort_time=True)
    init_file = 5
    final_file = len(files)
    n_file = final_file-init_file

    # Pull out attributes common to all files
    obj0 = hsDat(files[init_file])
    t0 = obj0.attribs['time']

    nsamp = obj0.attribs["nsamp"]
    fsamp = obj0.attribs["fsamp"]

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    upper2 = (2.0 / fsamp) * (f_rot + 0.5 * bw_filt)
    lower2 = (2.0 / fsamp) * (f_rot - 0.5 * bw_filt)

    b2, a2 = signal.butter(3, [lower2, upper2], btype='bandpass')

    # Construct time and frequency arrays from sampling parameters
    tarr0 = np.linspace(0, (nsamp-1)/fsamp, nsamp)
    freqs = np.fft.rfftfreq(nsamp, d = 1./fsamp)

    # Define array to be filled as we process data
    times = np.zeros(n_file)
    phases = np.zeros(n_file)
    phase_errs = np.zeros(n_file)
    dphases = np.zeros(n_file)
    dphase_errs = np.zeros(n_file)
    field_amps = np.zeros(n_file)
    field_amp_errs = np.zeros(n_file)
    field_freqs = np.zeros(n_file)
    field_freq_errs = np.zeros(n_file)
    pressures = np.zeros((n_file, 3))
    pressures_mbar = np.zeros((n_file, 3))

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

            elec_mon = obj.dat[:,drive_ax]
            drive_fft = np.fft.rfft(elec_mon)

            elec_filt = tabor_mon_fac * signal.filtfilt(b2, a2, elec_mon)

            zeros = np.zeros(nsamp)
            voltage = np.array([zeros, zeros, zeros, elec_filt, \
                       -elec_filt, zeros, zeros, zeros])
            efield = bu.trap_efield(voltage)

            # plt.loglog(freqs, np.abs(np.fft.rfft(elec_mon)))
            # plt.loglog(freqs, np.abs(np.fft.rfft(elec_filt)))
            # # for i in [0,1,2]:
            # #     plt.plot(voltage[3][:50000]-voltage[4][:50000])
            # plt.show()

            max_ind = np.argmax(np.abs(drive_fft))
            freq_guess = freqs[max_ind]
            phase_guess = np.mean(np.angle(drive_fft[max_ind-2:max_ind+2]))
            amp_guess = np.sqrt(2) * np.std(efield[0])
            p0 = [amp_guess, freq_guess, phase_guess, 0]

            fit_ind = int(0.01 * len(time_vec))
            popt, pcov = opti.curve_fit(sine, time_vec[:fit_ind], efield[0][:fit_ind], p0=p0)
            amp_fit = popt[0]
            amp_err = np.sqrt(pcov[0,0])

            field_amps[i] = amp_fit
            field_amp_errs[i] = amp_err

            field_freqs[i] = popt[1]
            field_freq_errs[i] = np.sqrt(pcov[1,1])

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
            phase_errs[i] = np.std(np.angle(dat_fft))
            dphases[i] = np.angle(np.sum(drive_fft))
            dphase_errs[i] = np.std(np.angle(drive_fft))
            # Convert raw pressure in torr to mbar
            pressures[i, :] = obj.attribs["pressures"]
            pressures_mbar[i, :] = np.array(obj.attribs["pressures"]) * 1.333
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
    delta_phi_err = np.sqrt(phase_errs**2 + 4.*dphase_errs**2)

    # Put all the phase between pi and -pi for better unwrapping 
    # later when we do more analysis
    delta_phi[delta_phi > np.pi] -= 2.*np.pi
    delta_phi[delta_phi < -np.pi] += 2.*np.pi
    delta_phi_err[delta_phi > np.pi] -= 2.*np.pi
    delta_phi_err[delta_phi < -np.pi] += 2.*np.pi

    if save:
        np.save(out_f + '_phi.npy', 0.5 * delta_phi)
        np.save(out_f + '_phi_err.npy', 0.5 * delta_phi_err)
        np.save(out_f + '_pressures.npy', pressures)
        np.save(out_f + '_pressures_mbar.npy', pressures_mbar)
        np.save(out_f + '_field_amp.npy', field_amps)
        np.save(out_f + '_field_amp_err.npy', field_amp_errs)
        np.save(out_f + '_field_freq.npy', field_freqs)
        np.save(out_f + '_field_freq_err.npy', field_freq_errs)
        np.save(out_f + '_time.npy', times)

        out_dict = {'time': times, 'phi': 0.5*delta_phi, 'phi_err': 0.5*delta_phi_err, \
                    'pressure': pressures, 'pressure_mbar': pressures_mbar, \
                    'field_amp': field_amps, 'field_amp_err': field_amp_errs, \
                    'field_freq': field_freqs, 'field_freq_err': field_freq_errs}

        pickle.dump(out_dict, open(out_f + '_all.p', 'wb'))
