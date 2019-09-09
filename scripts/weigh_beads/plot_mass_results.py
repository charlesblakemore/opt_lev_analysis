import numpy as np
import matplotlib.pyplot as plt

import dill as pickle

plt.rcParams.update({'font.size': 14})



overall_mass_dict = pickle.load(open('./overall_masses.p', 'rb'))
allres_dict = pickle.load(open('./allres.p', 'rb'))

dates = overall_mass_dict.keys()
dates.sort()

newdates = []
for dateind, date in enumerate(dates):
    if int(date) > 20190200:
        break
    newdates.append(date)
dates = newdates

plot_ind = 0

f, axarr = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,2]}, sharex=True,
                        figsize=(6,4),dpi=200)
for dateind, date in enumerate(dates):

    if int(date) > 20190200:
        continue

    dat = np.array(allres_dict[date])
    overall = np.array(overall_mass_dict[date])

    ncond = dat.shape[0]
    ms = (dat[:,4] > 1.0) * 60 + (dat[:,4] <= 1.0) * 15

    marker = []
    edgecolors = []
    facecolors = []
    for cond in dat:
        if cond[3] > 0:
            facecolors.append('r')
            edgecolors.append('r')
        elif cond[3] <= 0:
            facecolors.append('k')
            edgecolors.append('k')

        if cond[4] > 1.0:
            facecolors[-1] = 'w'

    xvec = np.array(range(ncond)) + plot_ind
    
    axarr[0].errorbar(np.array(range(ncond)) + plot_ind, \
                      dat[:,0], yerr=np.sqrt(dat[:,1]**2 + dat[:,2]**2), \
                      fmt='', ls='none', color='k')

    xarr = np.array(range(ncond)) + plot_ind
    yarr = dat[:,0]
    for xp, yp, fc, ec in zip(xarr, yarr, facecolors, edgecolors):
        axarr[0].scatter([xp], [yp], s=20, marker='o', \
                         facecolors=fc, edgecolors=ec, zorder=20)

    #axarr[0].scatter(np.array(range(ncond)) + plot_ind, dat[:,0], \
    #                 s=15, marker=marker,color=colors, zorder=20)
    
    if date != dates[-1]:
        axarr[0].axvline(ncond+plot_ind, color='k', \
                         linestyle='--', linewidth=1, alpha=0.3)


    if dateind > len(dates) - 4:
        axarr[1].errorbar([plot_ind + 1], [overall[0]], \
                          yerr=[np.sqrt(np.sum(overall[1:]**2))], \
                          marker=(6,2,0), #'P', \
                          ms=10, ls='none', color='b')
        plot_str = 'No.%i' % int(dateind - (len(dates) - 4))
        #axarr[1].text(plot_ind-0.5, overall[0] - 2.5, plot_str, color='b', fontsize=12)
        axarr[1].text(plot_ind+1, 80, plot_str, color='b', fontsize=10, ha='center')
        print date, overall[0], np.sqrt(overall[1]**2 + overall[2]**2), overall[3]
    else:
        axarr[1].errorbar([plot_ind + 1], [overall[0]], \
                          yerr=[np.sqrt(np.sum(overall[1:]**2))], \
                          fmt='.-', ms=10, ls='none', color='k')



    plot_ind += (ncond + 1)



#axarr[1].set_xlabel('All Measurements (chronological)')
axarr[1].set_xlabel('Time (arb. units)')
axarr[0].set_ylabel('Mass (pg)')
axarr[1].set_ylabel('Mass (pg)')
axarr[0].set_yticks([78,81,84,87])
axarr[1].set_yticks([78,81,84,87])
axarr[0].tick_params(axis='x', which='both', \
                         bottom=False, top=False, labelbottom=False)
axarr[1].tick_params(axis='x', which='both', \
                         bottom=False, top=False, labelbottom=False)
axarr[0].set_ylim(76,88)
axarr[1].set_ylim(77.5,87.5)
plt.tight_layout()
f.savefig('/home/charles/plots/weigh_beads/mass_vs_time_all_v3.png')
f.savefig('/home/charles/plots/weigh_beads/mass_vs_time_all_v3.pdf')
f.savefig('/home/charles/plots/weigh_beads/mass_vs_time_all_v3.svg')
plt.show()



    




overall_mass = np.load('./overall_masses.npy')

rawdat = np.load('./allres.npy')
print rawdat.shape

dat = []
for row in rawdat:
    if row[0] == False:
        continue
    else:
        dat.append(row)
dat = np.array(dat)

rinds = range(rawdat.shape[0])
inds = range(dat.shape[0])

ms1 = np.zeros_like(rinds)
ms1 += (rawdat[:,4] > 1.0) * 50
ms1 += (rawdat[:,4] <= 1.0) * 20

color1 = []
for ind in rinds:
    if rawdat[ind,3] > 0:
        color1.append('r')
    elif rawdat[ind,3] <= 0:
        color1.append('k')

f, axarr = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,2]}, sharex=True,
                        figsize=(6,4),dpi=200)
#plt.figure(figsize=(6,4),dpi=200)
axarr[0].errorbar(rinds, rawdat[:,0], yerr=np.sqrt(rawdat[:,1]**2 + rawdat[:,2]**2), \
                  fmt='', ls='none', color='k')
axarr[0].scatter(rinds, rawdat[:,0], \
                 s=ms1, color=color1, zorder=20)
bead_inds = [0]
for ind in rinds:
    if rawdat[ind,0] < 50:
        bead_inds.append(ind+1)
        if ind != rinds[-1]:
            axarr[0].axvline(ind, color='k', linestyle='--', linewidth=1, alpha=0.3)
bead_inds.pop(-1)
axarr[1].errorbar(np.array(bead_inds) + 1, \
                  overall_mass[:,0], yerr=overall_mass[:,1], fmt='.-', \
                  ms=10, ls='none', color='k')
axarr[1].set_xlabel('All Measurements (chronological)')
axarr[0].set_ylabel('Mass [pg]')
axarr[1].set_ylabel('Mass [pg]')
axarr[0].set_yticks([78,81,84,87])
axarr[1].set_yticks([78, 82, 86])
axarr[0].tick_params(axis='x', which='both', \
                         bottom=False, top=False, labelbottom=False)
axarr[1].tick_params(axis='x', which='both', \
                         bottom=False, top=False, labelbottom=False)
axarr[0].set_ylim(76,88)
axarr[1].set_ylim(76,88)
plt.tight_layout()
plt.savefig('/home/charles/plots/weigh_beads/mass_vs_time.png')
plt.show()




#plt.hist(dat[:,0], label='All')
plt.hist(overall_mass[:,0], label='Measured Masses')
plt.show()





ms = np.zeros_like(inds)
ms += (dat[:,4] > 1.0) * 50
ms += (dat[:,4] <= 1.0) * 20

color = []
for ind in inds:
    if dat[ind,3] > 0:
        color.append('r')
    elif dat[ind,3] <= 0:
        color.append('k')


massvec = dat[:,0]
err_stat = dat[:,1]
err_sys = dat[:,2]
err_tot = np.sqrt(err_stat**2 + err_sys**2)


plt.figure(figsize=(6,4),dpi=200)
plt.errorbar(dat[:,3], massvec, xerr=np.zeros_like(dat[:,0]), yerr=err_tot, \
             fmt='', ls='none', color='k')
plt.scatter(dat[:,3], massvec, s=ms, color=color)
plt.xlabel('Charge State [e]')
plt.ylabel('Measured Mass [pg]')
plt.ylim(76,88)
plt.tight_layout()
plt.savefig('/home/charles/plots/weigh_beads/mass_vs_charge.png')
plt.show()




plt.figure(figsize=(6,4),dpi=200)
plt.errorbar(dat[:,5]*1e6, massvec, xerr=np.zeros_like(dat[:,0]), yerr=err_tot, \
             fmt='', ls='none', color='k')
plt.scatter(dat[:,5]*1e6, massvec, s=ms, color=color)
plt.xlabel('$\mathcal{P}_{det}$ [$\mu$W]')
plt.ylabel('Measured Mass [pg]')
plt.ylim(76,88)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('/home/charles/plots/weigh_beads/mass_vs_pow.png')
plt.show()



