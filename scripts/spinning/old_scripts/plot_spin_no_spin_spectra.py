import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import bead_util as bu
import scipy.signal as ss


path_0 = "/data/20181030/bead1/spinning_trans_data/0Hz_div4"
path_50k = "/data/20181030/bead1/spinning_trans_data/50kKz_div4"



files_0 = bu.find_all_fnames(path_0)
files_50k = bu.find_all_fnames(path_50k)

df_0 = bu.DataFile()
df_0.load(files_0[0])
df_0.diagonalize()
df_50k = bu.DataFile()
df_50k.load(files_50k[0])
df_50k.diagonalize()
ns_0 = np.shape(df_0.pos_data)[-1]
ns_50k = np.shape(df_50k.pos_data)[-1]

fft_0 = np.einsum('ij, i->ij', np.fft.rfft(ss.detrend(df_0.pos_data, axis = -1), axis = -1)*2./ns_0, df_0.conv_facs)

fft_50k = np.einsum('ij, i->ij', np.fft.rfft(ss.detrend(df_50k.pos_data, axis = -1), axis = -1)*2./ns_50k, df_50k.conv_facs)

freqs_0 = np.fft.rfftfreq(ns_0, d = 1./df_0.fsamp)
freqs_50k = np.fft.rfftfreq(ns_50k, d = 1./df_50k.fsamp)

ax_to_plot = 2
matplotlib.rcParams.update({'font.size':14})
f, ax = plt.subplots(dpi = 200)
ax.plot(freqs_0, np.abs(fft_0[ax_to_plot]), label = "0 Hz")
ax.plot(freqs_50k, np.abs(fft_50k[ax_to_plot]), label = "50 kHz")
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Force [N]")
plt.legend()
plt.show()

