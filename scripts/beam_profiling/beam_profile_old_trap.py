import numpy as np
import bead_util as bu

import matplotlib.pyplot as plt
import os, re
import scipy.signal as sig
import scipy
import glob

from scipy.optimize import curve_fit

from iminuit import Minuit

from bead_util import make_all_pardirs

plt.rcParams.update({'font.size': 14})


ysign = 0



# bu.configuration.col_labels["stage_pos"] = [17, 18, 19]
# data_dir1 = '/data/old_trap/20171026/profiling/xsweep_in2_5t_down1t_gf'
# data_dir2 = '/data/old_trap/20171026/profiling/ysweep_in2_5t_down1t'
# data_column1 = 5
# data_column2 = data_column1
# ysign = 1.0


# data_dir1 = '/data/old_trap/20180514/profiling/xsweep_final'
# data_dir2 = '/data/old_trap/20180514/profiling/ysweep_final'
# data_column1 = 3
# data_column2 = data_column1
# ysign = -1.0

# data_dir1 = '/data/old_trap/20190315/profiling/xsweep_adj13_atm'
# data_dir2 = '/data/old_trap/20190315/profiling/ysweep_right_adj13_atm'
# data_column1 = 1
# data_column2 = data_column1


# data_dir1 = '/data/old_trap/20190902/profiling/post_trans/xsweep_vac_init'
# data_dir2 = '/data/old_trap/20190902/profiling/post_trans/ysweep_vac_init'

# data_dir1 = '/data/old_trap/20190902/profiling/xsweep_vac_out3_0'
# data_dir2 = '/data/old_trap/20190902/profiling/ysweep_vac_in2_0'


# data_dir1 = '/data/old_trap/20201202/profiling/xsweep_init'
# data_dir2 = '/data/old_trap/20201202/profiling/xsweep_init'


# data_dir1 = '/data/old_trap/20210624/profiling/xsweep_centered'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_centered'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj1'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj4'
# data_dir1 = '/data/old_trap/20210624/profiling/xsweep_adj4'
# data_dir1 = '/data/old_trap/20210624/profiling/xsweep_adj8'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj11'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj12'
# data_dir1 = '/data/old_trap/20220810/beam_profiling/xsweep_y0um_3Hz_init'
# data_dir2 = '/data/old_trap/20220810/beam_profiling/ysweep_x30um_3_7Hz'

# data_dir2 = '/data/old_trap/20210628/profiling/ysweep_afterclean'
# data_dir2 = '/data/old_trap/20210628/profiling/ysweep_afterclean_nolights'
# data_dir2 = '/data/old_trap/20210628/profiling/ysweep_adj9'
# data_dir1 = '/data/old_trap/20210628/profiling/xsweep_adj10'
# data_dir1 = '/data/old_trap/20210628/profiling/xsweep_adj10'


# data_dir1 = '/data/old_trap/20230130/beam_profiling/xsweep_init'
# data_dir2 = '/data/old_trap/20230130/beam_profiling/ysweep_init'

# data_dir1 = '/data/old_trap/20230131/beam_profiling/xsweep_vert80um'
# data_dir2 = '/data/old_trap/20230131/beam_profiling/ysweep_vert80um'
# data_dir1 = '/data/old_trap/20230131/beam_profiling/xsweep_vert160um'
# data_dir2 = '/data/old_trap/20230131/beam_profiling/ysweep_vert160um'
# data_dir1 = '/data/old_trap/20230131/beam_profiling/xsweep_vert240um'
# data_dir2 = '/data/old_trap/20230131/beam_profiling/ysweep_vert240um'
# data_dir1 = '/data/old_trap/20230131/beam_profiling/xsweep_vert520um'
# data_dir2 = '/data/old_trap/20230131/beam_profiling/ysweep_vert520um'

# data_dir1 = '/data/old_trap/20230131/beam_profiling/ysweep_vert520um'
# data_dir2 = '/data/old_trap/20230201/beam_profiling/ysweep_vert520um_vangle0_25turn_in'
# data_dir1 = '/data/old_trap/20230201/beam_profiling/xsweep_vert520um_vangle1_25turn_in'
# data_dir2 = '/data/old_trap/20230201/beam_profiling/ysweep_vert520um_vangle1_25turn_in'
# data_dir2 = '/data/old_trap/20230201/beam_profiling/ysweep_vert520um_vangle1_25turn_in_ytrans_1turn_in'


# data_dir1 = '/data/old_trap/20230202/beam_profiling/ysweep_pos1_3_1Hz'
# data_dir2 = '/data/old_trap/20230202/beam_profiling/ysweep_pos1_11Hz'

# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/xsweep_no_coverslip'
# data_dir2 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip'

# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/xsweep_no_coverslip_down_0_5turn'
# data_dir2 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_0_5turn'

# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/xsweep_no_coverslip_down_1turn'
# data_dir2 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_1turn'

# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/xsweep_no_coverslip_down_1_75turn'
# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_1_75turn'
# data_dir2 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_1_75turn_vangle_out_1turn'
# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_1_75turn_vangle_in_0_5turn'
# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_1_75turn_vangle_in_0_75turn_vtrans_1turn'
# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/xsweep_no_coverslip_down_1_75turn_vangle_in_0_75turn_vtrans_down_2turn'
# data_dir2 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_1_75turn_vangle_in_0_75turn_vtrans_down_2turn'

# data_dir1 = '/data/old_trap/20230202/beam_profiling_2/xsweep_no_coverslip_down_2turn'
# data_dir2 = '/data/old_trap/20230202/beam_profiling_2/ysweep_no_coverslip_down_2turn'

# data_dir1 = '/data/old_trap/20230203/beam_profiling/ysweep_pos1'
# data_dir2 = '/data/old_trap/20230203/beam_profiling/ysweep_pos4'

# data_dir1 = '/data/old_trap/20230203/beam_profiling_2/xsweep_after_shimming'
# data_dir2 = '/data/old_trap/20230203/beam_profiling_2/ysweep_after_shimming'

# data_dir1 = '/data/old_trap/20230206/beam_profiling/xsweep_under_vacuum'
# data_dir2 = '/data/old_trap/20230206/beam_profiling/ysweep_under_vacuum'

data_dir1 = '/data/old_trap/20230322/beam_profiling/xsweep_init_17Hz'
data_dir2 = '/data/old_trap/20230322/beam_profiling/ysweep_init_17Hz'

show = True
savefigs = True
save_base = '/home/cblakemore/plots/beam_profiles/'

# identifier = data_dir1.split('sweep')[-1][1:]
# identifier = data_dir2.split('sweep')[-1][1:]
# identifier = 'vert80um'
# identifier = 'XY Astigmatism'
# identifier = 'different positions'

identifier = '- after power outage'



######################################################
######  BELOW VALUES MIGHT OCCASIONALLY CHANGE  ######
######################################################

stage_cols1 = [1, 2, 3]
stage_cols2 = [1, 2, 3]
data_column1 = 0
data_column2 = 0

ysign = 1.0

### For legacy data
# data_column1 = 1
# stage_cols1 = [2, 3, 4]

ideal_waist1 = 2.5   # um
ideal_waist2 = 2.5   # um

# debug_plot = True
debug_plot = False

multi_dir = True
# multi_dir = False

label_xy = True   ### Otherwise labels "before/after""
# opt_ext = ''
opt_ext = f' {identifier}'

waterfall_profs = True

height_to_plot = 0.

log_profs = True

ROI = [-80.0, 80.0] # um
#OFFSET = 2.*10**(-5)
OFFSET = 0

msq_fit = True
gauss_fit = True
# gauss_fit_limits = [-2.5, 2.5]

msq_with_gauss = True

vec_inds = (100, -100)

nbin = 200

sigmasq_cutoff = 1e-3

baseline_fit = True
baseline_edge = 30.0




######################################################
########  NOT MUCH SHOULD CHANGE BELOW HERE  #########
######################################################

date1 = re.search(r"\d{8,}", data_dir1)[0]
if multi_dir:
    date2 = re.search(r"\d{8,}", data_dir2)[0]
    if date1 != date2:
        date1 = date1 + '_vs_' + date2

def gaussian_1d(x, A, x0, sigmax, const):
    return A * np.exp( -1.0*(x-x0)**2 / (2.0*sigmax**2) )

def gaussian_waist_evolution(z, z0, w0, Msq, wavelength=1064.0e-9):
    return np.sqrt( w0**2 + Msq**2 * (wavelength / (np.pi * w0))**2 * (z - z0)**2 )

if savefigs:
    make_all_pardirs(os.path.join(save_base, date1, 'derp.file'))

    



def profile(fname, data_column=0, plot=False, nbin=200):
    df = bu.DataFile()
    df.load(fname, skip_fpga=False)
    df.load_other_data()
    df.calibrate_stage_position()

    if 'ysweep' in fname:
        stage_column = 1
    else:
        stage_column = 0

    dt = 1.0 / df.fsamp

    if plot:
        shape = np.shape(df.other_data)
        for i in range(shape[0]):
            plt.plot(df.other_data[i, :], label = str(i))
            plt.title('Data Columns')
        plt.legend(fontsize=10)
        plt.tight_layout()

        plt.figure()
        for j in range(3):
            plt.plot(df.cant_data[j,:], label=str(j))
            plt.title('Attractor Coordinates')
        plt.legend(fontsize=10)
        plt.tight_layout()

        plt.show()

        input()

    h = np.mean(df.cant_data[2, :])
    h_round = bu.round_sig(h, sig=3)

    if h_round < 10.0:
        h_round = bu.round_sig(h_round, sig=2)

    b, a = sig.butter(1, 0.5)
    int_filt = sig.filtfilt(b, a, df.other_data[data_column])
    proft = np.gradient(int_filt)

    # stage_filt = sig.filtfilt(b, a, df.cant_data[stage_column, :])
    # dir_sign = np.sign(np.gradient(stage_filt)) * sign

    dir_sign = np.sign(np.gradient(df.cant_data[stage_column]))

    # plt.plot(proft)
    # plt.plot(dir_sign)
    # plt.plot(proft*dir_sign)
    # plt.show()

    xvec = df.cant_data[stage_column, :]
    yvec = proft * dir_sign

    yvec_p = 0.5*(proft - proft * dir_sign)
    yvec_n = -0.5*(proft + proft * dir_sign)

    if np.sum(yvec) < 0:
        yvec *= -1.0

    # sort_inds = np.argsort(xvec)

    # b, y, e = bu.spatial_bin(xvec, yvec, dt, nbin=300, nharmonics=300, add_mean=True)
    b, y, e = bu.rebin(xvec[vec_inds[0]:vec_inds[1]], \
                       yvec[vec_inds[0]:vec_inds[1]], \
                       nbin=nbin, plot=False, correlated_errs=True)

    inds_p = yvec_p != 0.0
    inds_n = yvec_n != 0.0

    # # plt.scatter(xvec[vec_inds[0]:vec_inds[1]], \
    # #                    yvec[vec_inds[0]:vec_inds[1]], \
    # #                    alpha=0.01)
    # plt.scatter(xvec[vec_inds[0]:vec_inds[1]]*inds_p[vec_inds[0]:vec_inds[1]], \
    #                    yvec_p[vec_inds[0]:vec_inds[1]]*inds_p[vec_inds[0]:vec_inds[1]], \
    #                    marker='x', alpha=0.2, label='Positive')
    # plt.scatter(xvec[vec_inds[0]:vec_inds[1]]*inds_n[vec_inds[0]:vec_inds[1]], \
    #                    yvec_n[vec_inds[0]:vec_inds[1]]*inds_n[vec_inds[0]:vec_inds[1]], \
    #                    marker='+', alpha=0.2)
    # plt.scatter(b, y)

    # plt.xlabel('Knife-edge Position [$\\mu$m]')
    # plt.ylabel('Marginalized Irradiance [~W/m]')
    # plt.title(identifier, fontsize=14)

    # plt.tight_layout()
    # plt.show()

    return b, y, e, h_round



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

    def sigsq(self, cutoff=1e-9):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()

        derp1 = self.bins > ROI[0]
        derp2 = self.bins < ROI[1]
        derp3 = self.y > cutoff * np.max(self.y)

        ROIbool = derp1 * derp2 * derp3

        num = np.sum(self.bins[ROIbool]**2 * self.y[ROIbool] * self.dxs[ROIbool])
        denom = np.sum(self.y[ROIbool]*self.dxs[ROIbool])
        #norm = np.sum(self.y*self.dxs)
        self.sigmasq = num / denom
        #self.sigmasq = np.sum(self.bins**2*self.y)/norm
         

def proc_dir(dir, data_column=0, plot=False, stage_cols=[1,2,3]):

    bu.configuration.col_labels["stage_pos"] = stage_cols

    files, lengths = bu.find_all_fnames(dir)
    file_profs = []
    hs = []
    for fi in files:
        b, y, e, h = profile(fi, nbin=nbin, plot=plot, data_column=data_column)
        #print h
        if h not in hs:
            #if new height then create new profile object
            hs.append(h)
            f = File_prof(b, y, e, h)
            f.date = re.search(r"\d{8,}", dir)[0]
            file_profs.append(f)
        else:
            #if height repeated then append data to object for that height
            for fp in file_profs:
                if fp.cant_height == h:
                    fp.bins = np.append(fp.bins, b)
                    fp.y = np.append(fp.y, y)
                    fp.errors = np.append(fp.errors, e)
            
    #now rebin all profiles
    for fp in file_profs:

        sorter = np.argsort(fp.bins)
        fp.bins = fp.bins[sorter]
        fp.y = fp.y[sorter]
        fp.errors = fp.errors[sorter]
        fp.dxs = np.append(np.diff(fp.bins), 0)

        # if len(fp.bins) > nbin:
        #     b, y, e = bu.rebin(fp.bins, fp.y, nbin=nbin, correlated_errs=True)
        #     fp.bins = b
        #     fp.y = y
        #     fp.errors = e
        #     fp.dxs = np.append(np.diff(b), 0)#0 pad left trapizoid rule

    sigsq1 = []
    hs = []

    for f in file_profs:

        f.sigsq(cutoff=sigmasq_cutoff)
        sigsq1.append(f.sigmasq)
        hs.append(f.cant_height)

        if baseline_fit:
            fit_inds = np.abs(f.bins) > baseline_edge
            baseline = np.mean( (f.y)[fit_inds] )
            f.y -= baseline
        
    return file_profs, np.array(hs), np.array(sigsq1)


 
def plot_profs(fp_arr, title='', show=True, save=False):
    #plots average profile from different heights
    i = 1
    colors = bu.get_colormap(len(fp_arr), cmap='plasma')

    fp_arr_sort = sorted(fp_arr, key = lambda fp: fp.cant_height)

    fac = 1.0; offset = 0.0
    if waterfall_profs:
        if log_profs:
            fac = 10.0
        else:
            offset = 1.0

    max_val = 0.0
    for fp in fp_arr_sort:
        new_max_val = np.max(fp.y)
        if new_max_val > max_val:
            max_val = new_max_val

    fig1 = plt.figure()
    ax1 = fig1.gca()
    for fp_ind, fp in enumerate(fp_arr_sort):
        color = colors[fp_ind]
        #plt.errorbar(fp.bins, fp.y, fp.errors, label = str(np.round(fp.cant_height)) + 'um')
        lab = str(np.round(fp.cant_height)) + 'um'
        if multi_dir:
            ax1.plot(fp.bins, (fac**fp_ind)*(fp.y / max_val) + offset*fp_ind, \
                     'o', label = lab, color=color)
            ax1.set_ylabel("Marginalized Irradiance [arb]")
        else:
            ax1.plot(fp.bins, (fac**fp_ind)*fp.y + offset, \
                     'o', label = lab, color=color)
            ax1.set_ylabel("Marginalized Irradiance [~W/m]")
    ax1.set_xlabel("Knife-edge Position [$\\mu$m]")
    if log_profs:
        ax1.set_yscale('log')
        if multi_dir:
            ax1.set_ylim(10**(-5), 2.0 * fac**(len(fp_arr_sort)-1))
    else:
        ax1.set_yscale('linear')
    ax1.legend(loc='lower left', title='Axial\ncoordinate:', 
               ncol=1, title_fontsize=10, fontsize=10)
    if title:
        ax1.set_title(title)
    fig1.tight_layout()

    heights = []
    means = []
    for fp_ind, fp in enumerate(fp_arr_sort):
        heights.append(fp.cant_height)
        means.append(fp.mean)

    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(heights, means, 'o', ms=10)
    ax2.set_xlabel("Knife-edge Height [$\\mu$m]")
    ax2.set_ylabel("Beam centroid [$\\mu$m]")
    if title:
        ax2.set_title(title)
    fig2.tight_layout()

    if save:
        if len(title):
            figtype1 = f'{title.replace(" ", "_").lower()}_profiles'
            figtype2 = f'{title.replace(" ", "_").lower()}_centroids'
        else:
            figtype1 = f'{identifier.replace(" ", "_").lower()}_profiles'
            figtype2 = f'{identifier.replace(" ", "_").lower()}_centroids'

        if log_profs:
            figtype1 += '_log';

        figname1 = os.path.join(save_base, date1, figtype1 + '.svg')
        figname2 = os.path.join(save_base, date1, figtype2 + '.svg')
        print()
        print('Saving figures to:')
        print('    ' + figname1)
        print('    ' + figname2)

        fig1.savefig(figname1)
        fig2.savefig(figname2)

    if show:
        plt.show()



file_profs, hs, sigsq1 = proc_dir(data_dir1, data_column=data_column1, \
                                    plot=debug_plot, \
                                    stage_cols=stage_cols1)

show_prof = False
if show and not msq_fit and not gauss_fit:
    show_prof = True

if multi_dir:
    if label_xy:
        title1 = 'Xsweep'; title2 = 'Ysweep'
        title3 = 'x_vs_y'
    else:
        title1 = 'Before'; title2 = 'After'
        title3 = 'before_vs_after'
    title1 += opt_ext; title2 += opt_ext; title3 += opt_ext

    fp2, hs2, sigsq2 = proc_dir(data_dir2, data_column=data_column2, \
                                plot=debug_plot, \
                                stage_cols=stage_cols2)
    ind = np.argmin(np.abs(hs - height_to_plot))
    ind2 = np.argmin(np.abs(hs2 - height_to_plot))
    # plot_profs([file_profs[ind]] + [fp2[ind2]])
    plot_profs(file_profs, title=title1, show=show_prof, save=savefigs)
    plot_profs(fp2, title=title2, show=show_prof, save=savefigs)

else:
    if label_xy:
        if 'xsweep' in data_dir1:
            title3 = 'Xsweep'
        else:
            title3 = 'Ysweep'
        title3 += opt_ext
    else:
        title3 = opt_ext.strip()

    plot_profs(file_profs, title=title3, show=show_prof, save=savefigs)

if msq_fit:

    p0 = [5., 2.0, 40.]

    bfit = hs < 140.
    hplt = np.arange(np.min(hs), np.max(hs), 0.1)

    axial_positions = hs[bfit]
    waists = 2.0*np.sqrt(sigsq1[bfit])

    npts = len(waists)
    def cost(z0, w0, Msq):
        func_val = gaussian_waist_evolution(axial_positions*1e-6, \
                                            z0*1e-6, w0*1e-6, Msq)
        resid = (waists - func_val*1e6)**2
        return (1.0 / (npts - 1.0)) * np.sum(resid)

    m = Minuit(cost, \
               z0 = axial_positions[np.argmin(waists)], \
               w0 = np.min(waists), \
               Msq = 1.11, \
              )

    m.limits['z0'] = (-1.0*np.inf, np.inf)
    m.limits['w0'] = (0.0, np.max(waists))
    m.limits['Msq'] = (1.0, np.inf)

    m.errordef = 1
    m.print_level = 0

    result = m.migrad(ncall=10000)
    
    if multi_dir:
        bfit2 = hs2 < 140.
        hplt2 = np.arange(np.min(hs2), np.max(hs2), 0.1)

        axial_positions2 = hs2[bfit2]
        waists2 = 2.0*np.sqrt(sigsq2[bfit2])

        npts2 = len(waists2)
        def cost2(z0, w0, Msq):
            func_val = gaussian_waist_evolution(axial_positions2*1e-6, \
                                                z0*1e-6, w0*1e-6, Msq)
            resid = (waists2 - func_val*1e6)**2
            return (1.0 / (npts - 1.0)) * np.sum(resid)

        m2 = Minuit(cost2, \
                    z0 = axial_positions2[np.argmin(waists2)], \
                    w0 = np.min(waists2), \
                    Msq = 1.1, \
                   )

        m2.limits['z0'] = (-1.0*np.inf, np.inf)
        m2.limits['w0'] = (0.0, np.max(waists2))
        m2.limits['Msq'] = (1.0, np.inf)

        m2.errordef = 1
        m2.print_level = 0

        result2 = m2.migrad(ncall=10000)

        fig, axarr = plt.subplots(2, sharex=True, sharey=True, \
                                  figsize=(6,6))
        ax1 = axarr[0]
        ax2 = axarr[1]

    else:
        fig, axarr = plt.subplots(1, sharex=True)
        ax1 = axarr

        
    if multi_dir:    
        maxwaist = np.max(np.concatenate((waists, waists2)))
    else:
        maxwaist = np.max(waists)
    
    n_ideal = 100
    fit_colors = bu.get_colormap(n_ideal, bu.truncate_colormap('inferno', vmax=0.5), \
                                 buffer=False, invert=True)
    waist_arr1 = np.linspace(ideal_waist1, m.values['w0'], n_ideal)

    ax1.plot(hs, 2.0*np.sqrt(sigsq1), 'o')
    ax1.plot(hplt, gaussian_waist_evolution(hplt*1e-6, \
                                            m.values['z0']*1e-6, \
                                            m.values['w0']*1e-6, \
                                            m.values['Msq'],\
                                           )*1e6, \
             'r', linewidth=2, label="$M^2={:0.2g}$".format(m.values['Msq']))
    ax1.plot(hplt, gaussian_waist_evolution(hplt*1e-6, \
                                            m.values['z0']*1e-6, \
                                            ideal_waist1*1e-6, \
                                            1.0,\
                                           )*1e6, \
             'r', linewidth=2, ls='--', alpha=0.6)

    # ax1.plot(hplt, Szsq(hplt, *popt), color=fit_colors[0], linewidth=2, \
    #          label="$M^2={:0.2g}$".format(popt[1]**2))
    # for ideal_ind, color in enumerate(fit_colors):
    #     if ideal_ind == n_ideal - 1:
    #         break
    #     popt_ideal = np.copy(popt)
    #     popt_ideal[0] = waist_arr1[ideal_ind] / 2.0
    #     popt_ideal[1] = 1.0
    #     ax1.plot(hplt, Szsq(hplt, *popt_ideal), linewidth=2, \
    #              color=bu.lighten_color(fit_colors[ideal_ind], 1.0))

    ax1.set_title("Focus at $h = {:0.3g}~\\mu$m, 'Waist' $W_0 = {:0.2g}~\\mu$m"\
                        .format(m.values['z0'], m.values['w0']), fontsize=14)
    ax1.set_ylabel("$w(z)$ from 2$^{\\rm nd}$-moment [$\\mu{\\rm m}$]")
    ax1.set_ylim(0, maxwaist*1.2)
    ax1.legend(loc='lower left', fontsize=10)

    if multi_dir:

        waist_arr2 = np.linspace(ideal_waist2, m2.values['w0'], 100)

        ax2.plot(hs2, 2.0*np.sqrt(sigsq2), 'o')
        ax2.plot(hplt2, gaussian_waist_evolution(hplt2*1e-6, \
                                                 m2.values['z0']*1e-6, \
                                                 m2.values['w0']*1e-6, \
                                                 m2.values['Msq'],\
                                                )*1e6, \
                 'r', linewidth=2, label="$M^2={:0.2g}$".format(m2.values['Msq']))
        ax2.plot(hplt2, gaussian_waist_evolution(hplt2*1e-6, \
                                                 m2.values['z0']*1e-6, \
                                                 ideal_waist2*1e-6, \
                                                 1.0,\
                                                )*1e6, \
                 'r', linewidth=2, ls='--', alpha=0.6)

        ax2.set_title("Focus at $h = {:0.3g}~\\mu$m, 'Waist' $W_0 = {:0.2g}~\\mu$m"\
                            .format(m2.values['z0'], m2.values['w0']), fontsize=14)
        if label_xy:
            ax1.set_ylabel("$w_x(z)$ [$\\mu{\\rm m}$]")
            ax2.set_ylabel("$w_y(z)$ [$\\mu{\\rm m}$]")
        else:
            ax1.set_ylabel("$w_{\\rm before}$ [$\\mu{\\rm m}$]")
            ax2.set_ylabel("$w_{\\rm after}$ [$\\mu{\\rm m}$]")
        ax2.set_xlabel("Knife-edge height along optical axis [$\\mu$m]")
        ax2.set_ylim(0, maxwaist*1.2)
        ax2.legend(loc='lower left', fontsize=10)

    else:
        ax1.set_xlabel("Knife-edge height along optical axis [$\\mu$m]")

    fig.tight_layout()

    if savefigs:
        figtype = f'{title3.replace(" ", "_").lower()}_second_moment'
        figname = os.path.join(save_base, date1, figtype + '.svg')

        print()
        print('Saving figure to:')
        print('    ' + figname)

        fig.savefig(figname)



if gauss_fit:

    if multi_dir:
        f2, axarr2 = plt.subplots(2, sharex=True, sharey=True, 
                                  figsize=(6,6))
        ax1 = axarr2[0]
        ax2 = axarr2[1]
    else:
        f2, axarr2 = plt.subplots(1, sharex=True)
        ax1 = axarr2
    
    if msq_fit:
        bestfit = np.argmin(np.abs(np.array(hs) - m.values['z0']))
        if multi_dir:
            bestfit2 = np.argmin(np.abs(np.array(hs2) - m2.values['z0']))
    else:
        bestfit = 0
        if multi_dir:
            bestfit2 = 0
    
    lab = hs[bestfit]
    if multi_dir:
        lab2 = hs2[bestfit2]
                        
    bestprof = file_profs[bestfit]
    if multi_dir:
        bestprof2 = fp2[bestfit2]

    #p02 = [10**(-3), 0, 10, 10**(-7)]
    #popt2, pcov2 = curve_fit(gauss_wconst, bestprof.bins, bestprof.y, p0=p02)

    fit_func = lambda x, A, x0, sigmax: gaussian_1d(x, A, x0, sigmax, 0)

    p02 = [10**(-3), 0, 10]    
    popt3, pcov3 = curve_fit(fit_func, bestprof.bins, bestprof.y, p0=p02)
    fitpts = np.arange(np.min(bestprof.bins), np.max(bestprof.bins), 0.1)
    
    bestprof.y *= (1.0 / popt3[0])
    popt3[0] = 1.0

    if multi_dir:
        popt4, pcov4 = curve_fit(fit_func, bestprof2.bins, bestprof2.y, p0=p02)
        fitpts2 = np.arange(np.min(bestprof2.bins), np.max(bestprof2.bins), 0.1)

        bestprof2.y *= (1.0 / popt4[0])
        popt4[0] = 1.0

    
    lab1 = '$w_0 = {:0.2g}~\\mu$m'.format(2.0*np.abs(popt3[2])) 

    ax1.plot(bestprof.bins, bestprof.y, 'o')
    ax1.plot(fitpts, fit_func(fitpts, *popt3), 'r', linewidth = 2, label = lab1)

    # data_int = np.sum(bestprof.y) * (bestprof.bins[1] - bestprof.bins[0])
    # gauss_int = np.sum(fit_func(fitpts, *popt3)) * (fitpts[1] - fitpts[0])

    # print() 
    # print("Non-Gaussian Part (1): ", (data_int - gauss_int) / data_int)

    ax1.set_title(f'Gaussian fit at $z={hs[bestfit]:0.2g}~\\mu$m', fontsize=14 )
    ax1.set_ylabel("$d \\mathcal{P} / dx$ [arb.]")

    if log_profs:
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')

    ax1.set_ylim(popt3[0] * 1e-4, popt3[0] * 1.5)
    # ax1.set_ylim(10**(-6), popt3[0] * 10)
    ax1.legend(loc=0, fontsize=10)
    
    if multi_dir:
        lab2 = '$w_0 = {:0.2g}~\\mu$m'.format(2.0*np.abs(popt4[2]))

        ax2.plot(bestprof2.bins, bestprof2.y, 'o')
        ax2.plot(fitpts2, fit_func(fitpts2, *popt4), 'r', linewidth = 2, label=lab2)
    
        # data_int2 = np.sum(bestprof2.y) * (bestprof2.bins[1] - bestprof2.bins[0])
        # gauss_int2 = np.sum(fit_func(fitpts2, *popt4)) * (fitpts2[1] - fitpts2[0])
        # print("Non-Gaussian Part (2): ", (data_int2 - gauss_int2) / data_int2)

        ax2.set_title(f'Gaussian fit at $z={hs2[bestfit2]:0.2g}~\\mu$m', fontsize=14 )
        ax2.set_xlabel("Knife-edge position along beam [$\\mu$m]")
        if label_xy:
            ax2.set_ylabel("$d \\mathcal{P} / dy$ [arb.]")
        else:
            ax1.set_ylabel("$\\left(d\\mathcal{P}/dx\\right)_{\\rm before}$ [arb.]")
            ax2.set_ylabel("$\\left(d\\mathcal{P}/dx\\right)_{\\rm after}$ [arb.]")

        if log_profs:
            ax2.set_yscale('log')
        else:
            ax2.set_yscale('linear')

        ax2.set_ylim(popt4[0] * 1e-4, popt4[0] * 1.5)
        # ax2.set_ylim(10**(-6), popt4[0] * 10)
        ax2.legend(loc=0, fontsize=10)

    else:
        ax1.set_xlabel("Knife-edge position along beam [$\\mu$m]")
    
    f2.tight_layout()

    if savefigs:
        figtype = f'{title3.replace(" ", "_").lower()}_guassian_fits'
        if log_profs:
            figtype += '_log'

        figname = os.path.join(save_base, date1, figtype + '.svg')
        
        print()
        print('Saving figure to:')
        print('    ' + figname)

        f2.savefig(figname)



if show and (gauss_fit or msq_fit):

    plt.show()


