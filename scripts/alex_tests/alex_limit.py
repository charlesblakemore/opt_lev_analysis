import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os
import glob
import matplotlib.mlab as ml
import alex_test as at
exec(compile(open('microgravity/alpha_lambda_test.py').read(), 'microgravity/alpha_lambda_test.py', 'exec'))


dat_dir = "/data/20180308/bead2/grav_data/onepos_long"
files = bu.sort_files_by_timestamp(glob.glob(dat_dir + "/*.h5"))

#get the data
n_harms = 10
d_freq = 17.
dir_ind = 0
df = bu.DataFile()
df.load(files[0])
df.calibrate_stage_position()
df.diagonalize()
cf = df.conv_facs
ts = df.pos_data[dir_ind, :]*cf[dir_ind]
amps, phis, sigas, sigphis = at.get_amp_phase(ts, d_freq, n_harms)

#get template
fx = at.generate_template(df, yukfuncs[dir_ind][lam25umind])
temp_amps, temp_phis, temp_sigas, temp_sigphis = at.get_amp_phase(fx, d_freq, n_harms)


#fit data


c_coefs = amps*np.exp(1.j*phis)
temp_coefs = temp_amps*np.exp(1.j*temp_phis)
harms = np.arange(1, n_harms + 1)

temp_coefs = temp_amps*np.exp(1.j*temp_phis)

def ffun(alpha):
    error = alpha*10**10*temp_coefs - c_coefs
    we = error*np.conj(error)/(amps**2*2)
    return np.real(np.sum(we)/(len(error) - 2))



plt.errorbar(harms, np.real(c_coefs), sigas/np.sqrt(2), fmt = 'o', ms = 7, label = "data real component")
plt.errorbar(harms, np.imag(c_coefs), sigas/np.sqrt(2), fmt = 'o', ms = 7, label = "data imaginary component")

plt.plot(harms, 10**10*np.real(temp_coefs), 'o', ms = 7, label = "template real component")

plt.plot(harms, 10**10*np.imag(temp_coefs),'o', ms = 7, label = "template imaginary component")
plt.xlabel('harmonic number')
plt.ylabel('component value [N/sqrt[Hz]]')
plt.legend()
plt.show()




