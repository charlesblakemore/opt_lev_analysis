import os, time

import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu


# dirname = '/data/20230306/bead2/discharge/'
dirname = '/data/old_trap/20200727/bead1/discharge/fine/'

bu.configuration.col_labels['electrodes'] = [0, 1, 2, 3, 4, 5, 6, 7]

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}

ts = 10.0


########

max_corr = []
inphase_corr = []

plt.ion()

fig, ax = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)
ax.plot(max_corr)
ax.plot(inphase_corr)

old_mrf = ''


while True:

    files, _ = bu.find_all_fnames(dirname, sort_time=True)

    try:
        mrf = files[-2]
    except:
        mrf = ''

    if mrf != old_mrf:

        df = bu.DataFile()
        df.load(mrf)

        drive = df.electrode_data[elec_ind]
        resp = df.pos_data[pos_ind]

        freqs = np.fft.rfftfreq(len(resp), d=1.0/df.fsamp)
        fft = np.fft.rfft(resp)
        dfft = np.fft.rfft(drive)

        amp = np.abs(fft)
        phase = np.angle(fft)

        damp = np.abs(dfft)
        dphase = np.angle(dfft)

        ind = np.argmax(damp[1:]) + 1

        drive_freq = freqs[ind]

        corr = amp[ind] / damp[ind]
        max_corr.append(corr)
        inphase_corr.append( (corr * np.exp( 1.0j * (phase[ind] - dphase[ind]) )).real )

        ax.cla()
        ax2.cla()

        ax.plot(max_corr)
        plt.pause(0.001)
        ax.plot(inphase_corr)
        plt.pause(0.001)

        ax2.loglog(freqs, damp)
        plt.pause(0.001)
        ax2.loglog(freqs, amp)
        plt.pause(0.001)

        plt.draw()

        old_mrf = mrf
        try:
            np.savetxt(os.path.join(dirname, "current_corr.txt"), \
                       inphase_corr[-1])
        except:
            print("Can't save current correlation to: ")
            print(f"    <{dirname}>")

    time.sleep(ts)

        
        

    
