import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
import scipy.optimize as opti
import scipy.interpolate as interp
import matplotlib

plt.rcParams.update({'font.size': 14})

#base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/pramp_data/" 
base_path = '/processed_data/spinning/pramp_data/'

#in_fs = ["50k_1vpp", "50k_2vpp", "50k_3vpp", "50k_4vpp", "50k_5vpp", "50k_6vpp", "50k_7vpp", "50k_8vpp"]
in_fs = ['20190514_Ar_50kHz_4Vpp_2']
title_str = 'Ar #2'

cal = 0.66

def get_delta_phi(fname):
    delta_phi = np.load(base_path + fname + "_phi.npy")
    return delta_phi

def get_pressure(fname):
    pressures = np.load(base_path + fname + "_pressures.npy")
    return pressures


def build_full_pressure(pressures, pirani_ind=0, highp_baratron_ind=2, \
                        baratron_ind=2, bara_lim=0.015, pirani_lim=5.0e-4, \
                        plot=False):

    inds = np.array(range(len(pressures[:,0])))

    pirani_p = pressures[:,pirani_ind]
    bara_p = pressures[:,baratron_ind]
    bara_p2 = pressures[:,highp_baratron_ind]

    bara_p_good = bara_p < bara_lim
    pirani_p_good = pirani_p > pirani_lim

    overlap = bara_p_good * pirani_p_good

    def line(x, a, b):
        return a * x + b

    Ndat = np.sum(overlap)

    bara_popt, bara_pcov = opti.curve_fit(line, inds[overlap], bara_p[overlap])
    pirani_popt, pirani_pcov = opti.curve_fit(line, inds[overlap], pirani_p[overlap])

    pirani_p = ((pirani_p - pirani_popt[1]) / pirani_popt[0]) * bara_popt[0] + bara_popt[1]

    if plot:
        plt.plot(inds[bara_p_good], bara_p[bara_p_good])
        plt.plot(inds[pirani_p_good], pirani_p[pirani_p_good])


        plt.show()

    pirani_p_bad = np.invert(pirani_p_good)
    bara_p_bad = np.invert(bara_p_good)

    low_p = bara_p[pirani_p_bad]
    high_p = pirani_p[bara_p_bad]

    avg_p = 0.5 * (pirani_p[overlap] + bara_p[overlap])
    total_p = np.concatenate((low_p, avg_p, high_p))

    return total_p





def build_full_pressure_2(pressures, pirani_ind=0, highp_baratron_ind=2, \
                          baratron_ind=2, bara_lim=0.015, pirani_lim=5.0e-4, \
                          highp_bara_lim=0.001, plot=False, use_highp_bara=False):

    inds = np.array(range(len(pressures[:,0])))

    pirani_p = pressures[:,pirani_ind]
    bara_p = pressures[:,baratron_ind]
    bara_p2 = pressures[:,highp_baratron_ind]

    low_p = bara_p
    if use_highp_bara:
        high_p = bara_p2
    else:
        high_p = pirani_p

    low_p_good = low_p < bara_lim
    if use_highp_bara:
        high_p_good = high_p > highp_bara_lim
    else:
        high_p_good = high_p > pirani_lim

    overlap = low_p_good * high_p_good

    high_p_bad = np.invert(high_p_good)
    low_p_bad = np.invert(low_p_good)

    if use_highp_bara:
        overlap_avg = 0.5 * (low_p[overlap] + high_p[overlap])
        fac1 = np.mean(overlap_avg / low_p[overlap])
        low_p = low_p * fac1
        fac2 = np.mean(overlap_avg / high_p[overlap])
        high_p = high_p * fac2

    else:
        fac = np.mean(low_p[overlap] / high_p[overlap])
        high_p = high_p * fac

    low_p_only = low_p[high_p_bad]
    high_p_only = high_p[low_p_bad]

    avg_p_only = 0.5 * (low_p[overlap] + high_p[overlap])
    total_p = np.concatenate((low_p_only, avg_p_only, high_p_only))

    pres_func = interp.interp1d(inds, total_p, kind='quadratic')

    #pres_func_2 = interp.splrep(inds, total_p, s=5e-4)
    pres_func_2 = interp.splrep(inds, total_p, s=12e-4)

    #return pres_func_2(inds)
    return pres_func(inds), interp.splev(inds, pres_func_2, der=0)







# Get raw phase difference at fundamental rotation freq
# from previously analyzed files.
phases = np.array(map(get_delta_phi, in_fs))
pressures = np.array(map(get_pressure, in_fs))

uphases_all = []
pressures_all = []
lock_lost_ind_all = []
for dir_ind in range(phases.shape[0]):

    pressures_real = build_full_pressure(pressures[dir_ind], plot=False)
    pressures_real_2, pressures_real_smooth = build_full_pressure_2(pressures[dir_ind], plot=False)

    t = np.array(range(len(pressures[dir_ind]))) * 2.0

    plt.plot(pressures_real_2, label='Raw: Pirani + Baratron')
    plt.plot(pressures_real_smooth, label='Smoothed')
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [mbar]')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    pressures_real = pressures_real_smooth

    phi0 = np.mean(phases[dir_ind][:10])

    # Find where we lose lock by looking fore sharp derivative
    raw_grad = np.gradient(np.unwrap(2.0 * phases[dir_ind]))

    plt.figure()
    plt.plot(pressures_real, raw_grad)
    #plt.figure()
    #plt.plot(pressures_real, phases[dir_ind])
    plt.show()

    raw_grad_init = np.std(raw_grad[:int(0.01*len(raw_grad))])
    bad_inds = np.array(range(len(raw_grad)))[np.abs(raw_grad) > 10 * raw_grad_init]
    
    for indind, ind in enumerate(bad_inds):
        if ind == bad_inds[-2]:
            lock_lost_ind = -1
            break
        delta = np.abs(ind - bad_inds[indind+1])
        if delta < 10:
            delta2 = np.abs(ind - bad_inds[indind+2])
            if delta2 < 10:
                lock_lost_ind = ind
                break

    #lock_lost_ind = bad_inds[0]
    lock_lost_ind_all.append(lock_lost_ind)

    # Reconstruct phase difference of fundamental rotation by 
    # unwrapping data prior to losing lock, then using the raw
    # data after losing lock
    uphases = np.unwrap(2.0*phases[dir_ind]) / 2.0

    offset = np.mean(uphases[:10])
    uphases -= offset

    uphases[lock_lost_ind:] = phases[dir_ind][lock_lost_ind:]


    sort_inds = np.argsort(pressures_real)

    pressures_real_sorted = pressures_real[sort_inds]
    uphases_sorted = uphases[sort_inds]

    pressures_all.append(pressures_real_sorted)
    uphases_all.append(uphases_sorted)

    plt.plot(pressures_real_sorted, uphases_sorted)
    plt.show()




def phi_ffun(p, k, phi0):
    return -1.*np.arcsin(p/k) + phi0



popts = []
pcovs = []


for ind, lock_ind in enumerate(lock_lost_ind_all):
    pressures = pressures_all[ind]
    uphases = uphases_all[ind]

    fit_pressures = pressures[:lock_ind]
    fit_uphases = uphases[:lock_ind]
    p0 = [pressures[lock_ind], 0]
    pphi, covphi = curve_fit(phi_ffun, fit_pressures, fit_uphases, p0 = p0, \
                             bounds=([0.01, -np.inf], [0.15, np.inf]), maxfev=10000)
    popts.append(pphi)
    pcovs.append(covphi)

    plot_pressures = np.linspace(0, pphi[0], 100)

    line_p = np.linspace(-1.0*np.max(pressures), 2*np.max(pressures), 100)

    lab_str = '$P_{\mathrm{max}}$: %0.3f mbar' % pphi[0]

    plt.scatter(pressures, uphases / np.pi)
    plt.plot(plot_pressures, phi_ffun(plot_pressures, *pphi) / np.pi, color='r', lw=2, \
             label=lab_str)
    plt.plot(line_p, np.ones(100) * (-0.5), '--', lw=3, color='k', alpha=0.5)
    plt.xlabel('Pressure [mbar]')
    plt.ylabel('Phase offset [$\pi$ rad]')
    plt.xlim(-0.05*pphi[0], 1.2*pphi[0])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.suptitle(title_str, fontsize=20)
    plt.subplots_adjust(top=0.90)
    plt.show()
    
    print 'init: ', p0[0]
    print 'fit: ', pphi[0]
    print








colors = ["b", "g", "c", "m", "y", "k"]
linestyles = [":", "-.", "--", "-"]
plt_inds = [0, 3, 7]
labels = ["8.25kV/m", "33.0kV/m", "66.0kV/m"]
axi = 1
matplotlib.rcParams.update({'font.size':14})
f, axarr = plt.subplots(len(plt_inds)+2, 1, figsize = (6,7.5), dpi = 100, sharex = True, gridspec_kw = {"height_ratios":[10, 10, 10, 1, 10]})
for i, ax in enumerate(axarr[:-2]):
    ind = plt_inds[i]
    bi = p_fits[ind]<p_maxs[ind]-0.001
    bic = np.logical_and(p_fits[ind]>p_maxs[ind]-0.001, p_fits[ind]<p_maxs[ind]+0.005)
    p_plot = np.linspace(0, popts[ind][0], 1000)
    ax.plot(p_fits[ind][bi], (phases[ind][bi]-popts[ind][-1])/np.pi, '.', color = 'C0')
    ax.plot(p_fits[ind][bic], (phases[ind][bic]-popts[ind][-1])/np.pi, 'o', color = 'C0', alpha = 0.25)
    ax.plot([popts[ind][0]], [-0.5], "D", markersize = 10, color = "C3")
    if ind == plt_inds[-1]:
        text_xpos = 0.012
    else:
        text_xpos = 0.008

    #ax.text(popts[ind][0]-text_xpos, -0.05, labels[i], fontsize = 12)
    ax.text(0.06, -0.1, labels[i], fontsize = 12)
    ax.axhline(y = -0.5, linestyle = '--', color = 'k', alpha = 0.5)
    ax.plot(p_plot, (phi_ffun(p_plot, *popts[ind])-popts[ind][-1])/np.pi, 'r')
    ax.set_ylim([-0.6, 0.1])
    ax.set_xlim([-0.01, 0.11])
    ax.set_yticks([0, -0.25, -0.5])
    ax.legend()
    if i==axi:
        ax.set_ylabel(r"$\phi_{eq}$ $[\pi]$")


def line(x, m, b):
    return m*x + b

Es = np.array([1, 2, 3, 4, 5, 6, 7, 8])*cal*50./0.004
ps_plot = np.linspace(0, popts[-1][0], 1000)
popts = np.array(popts)
pcovs = np.array(pcovs)
scale = 1000
axarr[-2].axis("off")
popt, pcov = curve_fit(line, popts[:, 0], Es)
axarr[-1].plot(popts[plt_inds, 0], Es[plt_inds]/scale, "D", markersize = 10, color = "C3")
axarr[-1].plot(popts[:, 0], Es/scale, 'o', color = "C2")
axarr[-1].plot(ps_plot, line(ps_plot, *popt)/scale, 'r', label = r"$639 \pm 64$ (kV/m)/mbar")
axarr[-1].set_ylabel(r"$E$ [kV/m]")
axarr[-1].legend(loc = 4, fontsize = 12)
axarr[-1].set_ylim([0, 75])
plt.subplots_adjust(top = 0.96, bottom = 0.1, left = 0.18, right = 0.92, hspace = 0.3)


axarr[-3].set_xlabel("P [mbar]")
axarr[-3].xaxis.labelpad = 10
axarr[-1].yaxis.labelpad = 33
axarr[-1].set_xlabel("P$_{\pi/2}$ [mbar]")
#plt.ylabel(r"$\phi_{eq}$")
#plt.legend()
plt.show()
f.savefig("/home/arider/plots/20181221/phase_vs_pressure.png", dpi = 200)




























