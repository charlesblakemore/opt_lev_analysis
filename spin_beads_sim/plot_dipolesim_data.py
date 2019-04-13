
import sys
import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torsion_noise as tn


alldata = np.load('/spinsim_data/alldata_Vxy100Vrotchirp_1kHz.npy')
# alldata = np.load('./data/test_efield_out.npy')

# Time interval to plot since plotting all the data is way
# too many points
time_int = (249, 255)


time = alldata[:,0]
data = alldata[:,1:]

print data.shape
print time.shape

dt = time[1] - time[0]

start = np.argmin( np.abs(time - time_int[0]) )
stop = np.argmin( np.abs(time - time_int[1]) )

subt = time[start:stop]
subpoints = data[start:stop,:]

naxes = data.shape[1]
fig, axarr = plt.subplots(3,3,sharex='col',sharey='col')
for ax in [0,1,2]:
    axarr[ax,0].plot(subt, subpoints[:,ax])
    axarr[ax,1].plot(subt, subpoints[:,ax+3])
    axarr[ax,2].plot(subt, subpoints[:,ax+7])

plt.figure()
plt.plot(subt, subpoints[:,6])

'''

fig, axarr = plt.subplots(3,1,sharex=True,sharey=True,figsize=(7,9))

axarr_t = []
for ax in axarr:
    axarr_t.append(ax.twinx())

for ind in [0,1,2]: #,2,3]:
    plot_x, plot_t = signal.resample(subpoints[:,ind], int(len(subt) * 0.11), t=subt)
    plot_Ex, plot_Et = signal.resample(subpoints[:,ind+7], int(len(subt) * 0.11), t=subt)
    axarr[ind].plot(plot_t, plot_x)
    axarr_t[ind].plot(plot_Et, plot_Ex, color='k')   # The Efield

axarr[0].set_ylabel('px [C$\cdot$m]')
axarr[1].set_ylabel('py [C$\cdot$m]')
axarr[2].set_ylabel('pz [C$\cdot$m]')
axarr_t[0].set_ylabel('Ex [V/m]')
axarr_t[1].set_ylabel('Ey [V/m]')
axarr_t[2].set_ylabel('Ez [V/m]')

# axarr[3].set_ylabel('ptot [C$\cdot$m]')
axarr[2].set_xlabel('Time [s]')


# axarr[3].plot(subt, np.sqrt(subpoints[:,0]**2 + subpoints[:,1]**2 + subpoints[:,2]**2))
# axarr[3].plot(subt, (np.zeros_like(subt) + p0))

plt.tight_layout()


# fig2, axarr2 = plt.subplots(3,1,sharex=True,sharey=True,figsize=(7,9))

# tind = np.argmin( np.abs(time - 150) )
# for ind in [0,1,2]:
#     asd = np.abs(np.fft.rfft(data[:,ind][tind:]))
#     freqs = np.fft.rfftfreq(len(data[:,ind][tind:]), d=dt)
#     axarr2[ind].loglog(freqs, asd)

# axarr2[0].set_ylabel('px ASD [arb]')
# axarr2[1].set_ylabel('py ASD [arb]')
# axarr2[2].set_ylabel('pz ASD [arb]')
# axarr2[2].set_xlabel('Frequency [Hz]')

# plt.tight_layout()


fig3, ax = plt.subplots(1,1)
#ax.plot(time, 1 - energy_vec / energy_vec[0])
#ax.set_ylim([-1e-9, 1e-9])
plot_x, plot_t = signal.resample(subpoints[:,6], int(len(subt) * 0.05), t=subt)
ax.plot(plot_t, plot_x)
#ax.set_ylabel('Residual Energy [arb]')
ax.set_ylabel('Energy [J')
ax.set_xlabel('Time [s]')
plt.tight_layout()

'''

plt.show()






