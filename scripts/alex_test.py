import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os
import glob
import matplotlib.mlab as ml

dat_dir = "/data/20180308/bead2/grav_data/onepos_long"
fname = "turbombar_xyzcool_pumped5_stage-X7um-Y40um-Z8um_Ydrive40umAC-17Hz_0.h5"

def df_freq(df):
    '''returns the frequency vector for and fft of data'''
    n = len(df.pos_data[0])
    d = 1./df.fsamp
    freqs = np.fft.rfftfreq(n, d)
    return freqs

def drive_freq(df):
    '''determines cantilever drive frequency from the stage settings'''
    if df.stage_settings['x driven']:
        return df.stage_settings['x freq']
    if df.stage_settings['y driven']:
        return df.stage_settings['y freq']
    if df.stage_settings['z driven']:
        return df.stage_settings['z freq']

def get_drive_bin(df):
    '''returns the fft bin corresponding to the cantilefer drive frequency.'''
    drive_freq_Hz = drive_freq(df)
    freqs = df_freq(df)
    return np.argmin((freqs - drive_freq_Hz)**2)

def get_coef(df, n):
    '''gets component at fourier bin n'''
    return np.fft.rfft(df.diag_pos_data)[:, n]*(2./np.shape(df.pos_data)[1])


def get_harm_coef(df, nh = 1):
    '''returns the coefficient at nh harmonic cantilever drive frequency. 
       nh = 1 corresponds to the fundamental.'''
    n = get_drive_bin(df)
    coefs = np.zeros((3, nh), dtype = complex)
    for i in range(nh):
        coefs[:, i] = get_coef(df, n*i)
    return coefs

def estimate_noise(ts, n, hw = 20, make_plot = False, Fs = 5000):
    '''estimates the noise at bin n from the median 
       amplitude of the surrounding 2hW points.'''
    fft = np.fft.rfft(ts)*(np.sqrt(2.)/np.sqrt(len(ts)*Fs))
    lfft = fft[n-hw:n]
    rfft = fft[n+1:n+hw+1]
    if make_plot:
        freqs = np.fft.rfftfreq(len(ts), 1./Fs)
        plt.loglog(freqs, np.abs(fft))
        plt.loglog(freqs[n-hw:n+hw+1], np.abs(fft[n-hw:n+hw+1]), 'xr')
        plt.loglog(freqs[n], np.abs(fft[n]), 'xg', markersize = 10)
        plt.show()

    return np.median(np.abs(np.append(lfft, rfft)))

def generate_template(df, yukfunc, p0 = [60., 0., 10.]):
    '''given a data file generates a template of the expected 
       force in the time domain.'''
    #first get cantilever position vector in same coord system as 
    #numerical integration of attractor mass
    pvec = np.zeros_like(df.cant_data)
    pvec[0, :] = df.cant_data[0, :] - p0[0]
    pvec[1, :] = df.cant_data[1, :] - p0[1]
    pvec[2, :] = df.cant_data[2, :] - p0[2]
    pts = np.stack(pvec*1e-6, axis = -1)
    return yukfunc(pts)

def get_components(ts, f0, nh, Fs = 5000, make_plot = False):
    '''returns the real and imaginary components as well as the noise
       of the fft of ts at f0 up to nh harmonics'''
    freqs = np.fft.rfftfreq(len(ts), d = 1./Fs)
    bin0 = np.argmin((f0 - freqs)**2)
    harms = np.arange(1, nh + 1)*bin0
    fft = np.fft.rfft(ts)
    noise = np.zeros(nh)
    for i, n in enumerate(noise):
        noise[i] = estimate_noise(ts, harms[i])/np.sqrt(2)
    if make_plot:
        plt.loglog(freqs, np.abs(fft)**2)
        plt.loglog(freqs[harms], np.abs(fft[harms])**2, 'og')
        plt.show()
    return fft[harms], noise 

def get_amp_phase(ts, f0, nh, Fs = 5000, make_plot = False):
    '''Gets the amplitudes and phases of the first nh harmonics of 
       f0 in ts'''
    fft = np.fft.rfft(ml.detrend_linear(ts))*(np.sqrt(2.)/np.sqrt(len(ts)*Fs))
    freqs = np.fft.rfftfreq(len(ts), d = 1./Fs)
    b0 = np.argmin((freqs - f0)**2)
    harms = np.arange(1, nh+1)*b0
    amps = np.abs(fft[harms])
    phis = np.angle(fft[harms])
    sigas = np.zeros_like(amps)
    sigphis = np.zeros_like(amps)
    for i, h in enumerate(harms):
        sigas[i] = estimate_noise(ts, harms[i])
        sigphis[i] = sigas[i]/(np.abs(fft[h])*np.sqrt(2))
    if make_plot:
        plt.loglog(freqs, np.abs(fft))
        plt.plot(freqs[harms], np.abs(fft[harms]), 'og')
        plt.show()
    return amps, phis, sigas, sigphis



files = glob.glob(dat_dir + "/*.h5")
#preallocate memory

def proc_files(files, nh = 10, nfft = 25001): 
    nf = len(files)
    amparr = np.zeros((nf, nh))
    phiarr = np.zeros((nf, nh))
    sigasarr = np.zeros((nf, nh))
    sigphisarr = np.zeros((nf, nh))
    ave_fft = np.zeros(nfft, dtype = complex)
    for i, f in enumerate(files):
        df = bu.DataFile()
        df.load(f)
        df.calibrate_stage_position()
        df.diagonalize()
        cf = df.conv_facs
        ts = df.pos_data[0, :]*cf[0]
        amps, phis, sigas, sigphis = \
                get_amp_phase(ts, 17, nh)
        ave_fft += np.fft.rfft(ts)*np.sqrt(1./(nfft*5000.))
        amparr[i, :] = amps
        phiarr[i, :] = phis
        sigasarr[i, :] = sigas
        sigphisarr[i, :] = sigphis
    return amparr, phiarr, sigasarr, sigphisarr, ave_fft


def compute_phase(coefs):
    phis = np.angle(coefs[:, 0, :])
    sigs = np.sqrt(np.abs(coefs[:, 0, :]**-2)*coefs[:, 1, :]**2)
    return phis, sigs



