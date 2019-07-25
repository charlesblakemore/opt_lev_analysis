import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from scipy.optimize import curve_fit
import matplotlib

#base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/pramp_data/" 
#base_path = '/processed_data/spinning/pramp_data/'
#base_path = '/daq2/20190417/bead1/pramps/'
base_path = "/home/dmartin/analyzedData/20190514/pramp3/He"
#base_path = '/processed_data/spinning/pramp_data/he/'
#base_path = '/processed_data/spinning/pramp_data/'
#in_fs = #["50k_1vpp"],["50k_2vpp"]#, "50k_3vpp", "50k_4vpp", "50k_5vpp", "50k_6vpp", "50k_7vpp", "50k_8vpp"]i

#in_fs = ['50k_4vpp']
#in_fs = ['49k_200vpp']
#in_fs = ["20190417_He_"]#,"20190417_N2_"]
#in_fs = ["20190417_He_"]
#in_fs = ["He_3vpp_0_", "./N2/N2_3vpp_0_"]
#in_fs = ['50k_5vpp_he_2_']
#in_fs = ['20190507_N2_50kHz_4Vpp_2_']
#in_fs = ['20190507_N2_50kHz_4Vpp_']
#in_fs = ['20190514_N2_50kHz_4Vpp_2_']
in_fs = ['He_4Vpp_50kHz_1_']

cal = 0.66

def get_phi(fname):
    phi = np.load(base_path + fname + "phi.npy")
    return phi/2

def get_pressure(fname):
    pressures = np.load(base_path + fname + "pressures.npy")
    return pressures

def pressure_model(pressures, break_ind = 483 , p_ind = 0, plt_press = True):
    ffun = lambda x, y0, m0, m1,m2: \
            pw_line(x, break_ind, 1e8, 1e7, y0, m0, m1, m2, 0.)

    n = np.shape(pressures)[0]
    inds = np.arange(n)
    popt, pcov = curve_fit(ffun, inds, pressures[:, p_ind])
    pfit = ffun(inds, *popt)

    if plt_press:
        p_dict = {0:"Pirani",1:"1 Torr Baratron", -1:"Baratron"}
        plt.plot(pressures[:, p_ind], '-', label = p_dict[p_ind])
        plt.plot(pfit, label = "piecewise linear fit")
        plt.xlabel("File [#]")
        plt.ylabel("Pressure [mbar]")
        plt.legend()
        plt.show()

    return pfit

def constline(x,m,b):
	return m*x + b
def phi_ffun(p, k, phinot):
    return -1*np.arcsin(np.clip(p/k, 0., 1)) + phinot

phases = np.array(map(get_phi, in_fs))
pressures = np.array(map(get_pressure, in_fs))
p_fits = np.array(map(pressure_model, pressures))

pmax = 0.01
phases[(phases<0)] += np.pi
phases *= -1
#Masks are to restrict the pressures over which fitting occurs
mask = [(p_fits[0] < pmax)] 

for i in xrange(1):
	popt, pcov = curve_fit(phi_ffun, p_fits[i][mask[i]], phases[i][mask[i]],p0=[pmax,0])
	print(popt)
	
	f = plt.figure()	
	
	plt.subplot()

	plt.scatter(p_fits[i], (phases[i]-popt[1])/np.pi, alpha = 0.25)
	plt.plot(p_fits[i],(phi_ffun(p_fits[i],*popt)-popt[1])/np.pi, 'r')
	plt.plot(p_fits[i],constline(p_fits[i],0,-0.5),'r')
	plt.text(0.05,0.2,'$P_{{max, Kr}}$: {} mbar'.format(round(popt[0],4)))
	plt.xlabel('Pressure (mbar)')
	plt.ylabel('Phase [$\pi$]')
	
#	f.savefig("/home/dmartin/analyzedData/20190514/Xe_50kHz_4Vpp_1", dpi = 200)	

plt.show()

p_maxs = np.array([0.011, 0.024, 0.036, 0.048, 0.062, 0.074, 0.085, 0.1004]) - 0.001
popts = []
pcovs = []

#p0 = [0.02, 0.75]

for i, p in enumerate(p_fits):
    bfit = p<p_maxs[i]
    p0 = [p_maxs[i], 0.75]
    pphi, covphi = curve_fit(phi_ffun, p_fits[i][bfit], phases[i][bfit], p0 = p0)
    popts.append(pphi)
    pcovs.append(covphi)
    

'''
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
'''
#plt.ylabel(r"$\phi_{eq}$")
#plt.legend()
plt.show()
f.savefig("/home/dmartin/analyzedData/20190514/", dpi = 200)
