import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy


def sbin_pn(xvec, yvec, bin_size=1., vel_mult = 0.):
    #Bins yvec based on binning xvec into bin_size for velocities*vel_mult>0.
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals /= fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    if vel_mult:
        vb = np.gradient(xvec)*vel_mult>0.
        yvec2 = yvec[vb]
    else:
        vb = yvec == yvec
        yvec2 = yvec

    for i, b in enumerate(bins):
        idx = bins_vals[vb] == b
        y_binned[i] = np.mean(yvec2[idx])
        y_errors[i] = np.std(yvec2[idx])
    return bins, y_binned, y_errors


def fit_fun(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C

width = 1
nharmonics = 100
numbins = 100

# Generate a time array, cantilever drive and some noise
t = np.arange(0, 10, 1. / 5000)
noise = np.random.randn(len(t)) * 0.1

cant = 40 * np.sin(2 * np.pi * 13 * t) + 40
cant_n = cant + noise

freqs = np.fft.rfftfreq(len(cant), d=1./5000)
cantfft = np.fft.rfft(cant_n)

fund_ind = np.argmax( np.abs(cantfft[1:]) ) + 1
drive_freq = freqs[fund_ind]

p0 = [75, drive_freq, 0, 0]
popt, pcov = optimize.curve_fit(fit_fun, t, cant_n, p0=p0)

fitdat = fit_fun(t, *popt)
mindat = np.min(fitdat)
maxdat = np.max(fitdat)

posvec = np.linspace(mindat, maxdat, numbins)

points = np.linspace(mindat, maxdat, 10.0*numbins)
fcurve = 2 * np.cos(0.7*points)+2
#plt.plot(points, fcurve)
#plt.show()

lookup = interp.interp1d(points, fcurve, fill_value='extrapolate')

dat = lookup(cant)
dat_n = dat + noise

datfft = np.fft.rfft(dat_n)


bins, rdat, rerrs = sbin_pn(t, dat, bin_size=1.)

plt.plot(bins, rdat)


fftsq = np.abs(datfft)

#if noise:
#    cantfilt = (fftsq) / (fftsq[fund_ind])    # Normalize filter to 1 at fundamental
#elif not noise:

cantfilt = np.zeros(len(fftsq))
cantfilt[fund_ind] = 1.0

if width:
    lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
    upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
    cantfilt[lower_ind:upper_ind+1] = cantfilt[fund_ind]

#plt.figure()
#plt.loglog(self.fft_freqs, cantfilt)

# Make a list of the harmonics to include and remove the fundamental 
harms = np.array([x+2 for x in range(nharmonics)])    

for n in harms:
    harm_ind = np.argmin( np.abs(n * drive_freq - freqs))
    cantfilt[harm_ind] = cantfilt[fund_ind]
    if width:
        h_lower_ind = harm_ind - (fund_ind - lower_ind)
        h_upper_ind = harm_ind + (upper_ind - fund_ind)
        cantfilt[h_lower_ind:h_upper_ind+1] = cantfilt[harm_ind]

cantr = np.fft.irfft(cantfilt * cantfft)
datr = np.fft.irfft(cantfilt * datfft)
plt.plot(cantr, datr)


plt.figure()
plt.plot(t, cantr)
plt.plot(t, datr)

plt.show()

'''

# Make a filter

eigenvectors = []
eigenvectors.append([1, cantfft[fund_ind]]) 

if width:
    lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
    upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))


harms = np.array( [x+2 for x in range(nharmonics)] )

for n in harms:
    harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
    eigenvectors.append([n, datfft[harm_ind]])



#print eigenvectors

out = np.zeros(len(posvec))

for vec in eigenvectors:
    power = vec[0]

    amp = np.abs(vec[1]) / len(t)
    phase = np.angle(vec[1]) + 0.5 * np.pi

    #if (phase < -0.1 or phase > 0.1):
    #    amp *= -1.0

    newposvec = posvec
    out += amp * newposvec**power

plt.plot(posvec, out) 
plt.show()

'''
