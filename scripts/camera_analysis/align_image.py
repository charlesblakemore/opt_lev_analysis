import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit
    

data_dir1 = "/data/20170704/profiling/zsweep5"
data_dir2 = "/data/20170704/profiling/ysweep5"
out_dir = "/calibrations/20170704"


#stage x = col 17, stage y = 18, stage z = 19
stage_column_z = 19
stage_column_y = 18

data_column = 4
cant_cal = 8. #um/volt

ROI = [0., 80.]

def spatial_bin(xvec, yvec, bin_size = .13):
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals/=fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    for i, b in enumerate(bins):
        idx = bins_vals == b
        y_binned[i] =  np.mean(yvec[idx])
        y_errors[i] = scipy.stats.sem(yvec[idx])
    return bins, y_binned, y_errors
    
        
    

def profile(fname, ends = 100, stage_cal = 8.):
    dat, attribs, f = bu.getdata(fname)
    dat = dat[ends:-ends, :]
    if 'zsweep' in fname:
        stage_column = 19
    elif 'ysweep' in fname:
        stage_column = 18
    dat[:,stage_column]*=stage_cal
    h = attribs["stage_settings"][0]*cant_cal
    f.close()
    b, a = sig.butter(1, 1)
    int_filt = sig.filtfilt(b, a, dat[:, data_column])
    proft = np.gradient(int_filt)
    if 'zsweep' in fname:
        stage_filt = sig.filtfilt(b, a, dat[:, stage_column_z])
       
    elif 'ysweep' in fname:
        stage_filt = sig.filtfilt(b, a, dat[:, stage_column_y])
       
    dir_sign = np.sign(np.gradient(stage_filt))
    b, y, e = spatial_bin(dat[dir_sign<0, stage_column], proft[dir_sign<0])
    return b, y, e, h

class File_prof:
    "Class storing information from a single file"
    
    def __init__(self, b, y, e, h):
        self.bins = b
        self.dxs = np.append(np.diff(b), 0)#0 pad left trapizoid rule
        self.y = y
        self.errors = e
        self.cant_height = h
        self.mean = "mean not computed"
        self.sigmasq = "std dev not computed"
        self.date = "date not entered"
        
    def dist_mean(self):
        #Finds the cnetroid of intensity distribution. subtracts centroid from bins
        norm = np.sum(self.y*self.dxs)
        self.mean = np.sum(self.dxs*self.y*self.bins)/norm
        self.bins -= self.mean

    def sigsq(self):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()
        derp1 = self.bins > ROI[0]
        derp2 = self.bins < ROI[1]
        ROIbool = np.array([a and b for a, b in zip(derp1, derp2)])
        norm = np.sum(self.y[ROIbool]*self.dxs[ROIbool])
        #norm = np.sum(self.y*self.dxs)
        self.sigmasq = np.sum(self.bins[ROIbool]**2*self.y[ROIbool])/norm
        #self.sigmasq = np.sum(self.bins**2*self.y)/norm
         

def proc_dir(dir):
    files = glob.glob(dir + '/*.h5')
    file_profs = []
    hs = []
    for fi in files:
        b, y, e, h = profile(fi)
        if h not in hs:
            #if new height then create new profile object
            hs.append(h)
            f = File_prof(b, y, e, h)
            f.date = dir[8:16]
            file_profs.append(f)
        else:
            #if height repeated then append data to object for that height
            for fi in file_profs:
                if fi.cant_height == h:
                    fi.bins = np.append(fi.bins, b)
                    fi.y = np.append(fi.y, y)
                    fi.errors = np.append(fi.errors, e)
            
    #now rebin all profiles
    for fp in file_profs:
        b, y, e = spatial_bin(fp.bins, fp.y)
        fp.bins = b
        fp.y = y
        fp.errors = e
        fp.dxs = np.append(np.diff(fp.bins), 0)#0 pad left trapizoid rule

    sigmasqs = []
    hs = []

    for f in file_profs:
        f.sigsq()
        sigmasqs.append(f.sigmasq)
        hs.append(f.cant_height)
        
    return file_profs, np.array(hs), np.array(sigmasqs)
 
def plot_profs(fp_arr, log_profs = True, show = False, other_label = ''):
    #plots average profile from different heights
    i = 1
    for fp in fp_arr:
        if not other_label:
            lab = str(np.round(fp.cant_height)) + 'um'
        else:
            lab = other_label
        i += 1
        plt.plot(fp.bins, fp.y, 'o', label = lab)
    plt.xlabel("position [um]")
    plt.ylabel("margenalized irradiance ~[W/m]")
    if log_profs:
        plt.gca().set_yscale('log')
    else:
        plt.gca().set_yscale('linear')
    plt.legend()
    if show:
        plt.show()


def Szsq(z, s0, M, z0, lam = 1.064):
    #function giving propigation of W=2sig parameter. See Seegman
    W0 = 2.*s0
    Wzsq = W0**2 + M**4*(lam/(np.pi*W0))**2*(z-z0)**2
    return Wzsq/4.

def save_cal(p_arr, path):
    #Makes path if it does not exist and saves parr to path/stage_position.npy
    if not os.path.exists(path):
        os.makedirs(path)
    outfile = os.path.join(path, 'stage_position')
    np.save(outfile, p_arr)

file_profs_z, hs, sigmasqs = proc_dir(data_dir1)
file_profs_y, hs, sigmasqs = proc_dir(data_dir2)

p_arr = np.array([file_profs_z[0].bins[0], file_profs_y[0].bins[0]])
save_cal(p_arr, out_dir)


plot_profs(file_profs_z, other_label = "z profile")
plot_profs(file_profs_y, other_label = "y profile")
plt.show()

