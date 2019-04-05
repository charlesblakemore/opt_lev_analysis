import numpy as np
import matplotlib.pyplot as plt
import flywheel as fw
import bead_util as bu
import re

path = "/data/20180927/bead1/spinning/amp_ramp_10s"
files = bu.find_all_fnames(path)

freqs = np.fft.rfftfreq(50000, 1./5000.)
bw = 1.
axis = 0
d_freqs = []
r_amps = []
sr_amps = []
r_phis = []
sr_phis = []
times = []
d_amps = []
df = bu.DataFile()

for i, f in enumerate(files):
    df.load(f)
    df.load_other_data()
    ig = float(re.findall('\d+Hz', f)[0][:-2])
    ig_ind = np.argmin(np.abs(freqs - ig))
    di, d_amp = fw.get_drive_ind(df.other_data[2], ig_ind)
    dfreq = freqs[di]
    filt = np.abs(freqs-dfreq)<bw
    respft = np.fft.rfft(df.pos_data[axis])
    respft[np.logical_not(filt)] = 0.
    resp = np.fft.irfft(respft)
    a, p = fw.anal_signal(resp)
    popt = fw.get_drive_phase(df.other_data[2], end = -1, make_plot = False)
    d_phi = fw.line(np.arange(len(resp)), *popt)
    delta_phi = p-d_phi
    d_freqs.append(dfreq)
    r_amps.append(np.mean(a))
    sr_amps.append(np.std(a))
    r_phis.append(np.mean(delta_phi))
    sr_phis.append(np.std(delta_phi))
    times.append(df.time)
    d_amps.append(d_amp)
    print i



