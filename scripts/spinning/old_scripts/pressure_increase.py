import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import bead_util as bu
from scipy.optimize import curve_fit
import dill as pickle


cpath = "/data/20180927/bead1/spinning/1k_valve_of"
filesc = bu.find_all_fnames(cpath)


freqs = np.fft.rfftfreq(50000, d = 1./5000)




def get_dir_data(files, drive_range = [1., 1500.], drive_amp = 5000.):
    nf = len(files)
    ns = 50000
    nfreq = len(freqs)
    d_ave = np.zeros((nf, nfreq), dtype = complex)
    pb = []
    pc = []
    pp = []
    t = []
    xs = np.zeros((nf, nfreq), dtype = complex)
    ys = np.zeros((nf, nfreq), dtype = complex)
    zs = np.zeros((nf, nfreq), dtype = complex)
    for i, f in enumerate(files):
        df = bu.DataFile()
        df.load(f)
        df.load_other_data()
        drive = df.other_data[2]
        dbool = (np.abs(np.fft.rfft(drive))>drive_amp)*(freqs>drive_range[0])*(freqs<drive_range[1])
        if not np.sum(dbool):
            plt.loglog(freqs, np.abs(np.fft.rfft(drive)))
            plt.loglog(freqs[dbool], np.abs(np.fft.rfft(drive))[dbool], 'o')
            plt.show()
        phi = np.angle(np.mean(np.fft.rfft(drive)[dbool]))
        d_ave[i, :] = np.fft.rfft(drive)*np.exp(-1.j*phi)
        xs[i, :] = np.fft.rfft(df.pos_data[0])*np.exp(-1.j*phi)
        ys[i, :] = np.fft.rfft(df.pos_data[1])*np.exp(-1.j*phi)
        zs[i, :] = np.fft.rfft(df.pos_data[2])*np.exp(-1.j*phi)
        pb.append(df.pressures['baratron'])
        pc.append(df.pressures['cold_cathode'])
        pp.append(df.pressures['pirani'])
        t.append(df.time)
    return {'d':d_ave, 'x':xs, 'y':ys, 'z':zs, 'p':np.array([pb, pc, pp]), 't':t}



def mov_ave(arr, n = 100):
    s = np.shape(arr)
    n_ave = int(np.floor(s[1]/n))
    arr_out = np.zeros((n_ave, s[-1]), dtype = complex)
    for i in range(n_ave):
        arr_out[i, :] = np.mean(arr[i*n:(i+1)*n, :], axis = 0)
    return arr_out



fb = (freqs>1023.)*(freqs<1026.)


