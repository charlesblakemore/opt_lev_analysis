import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize
import cPickle as pickle
import time
from multiprocessing import Pool

####################################################
####### Input parameters for data processing #######


pressure_dir = pickle.load(open("pressure_results_dir.p", "rb"))

avg = True

iterations = ['o', 's', '^', '*']
colors = {'He': ('#ff751a',4), 'Ne': ('#fddc35',22), 'Ar': ('#0000ff',40), \
          'Kr': ('#00cc00',85), 'Xe': ('#bb33ff',131)}
masses = []
for color in colors:
    masses.append(colors[color][1])
masses.sort()
masses = np.array(masses)
deltapp_raw = np.zeros(len(masses))
deltapperrs_raw = np.zeros(len(masses))
deltapp_fit = np.zeros(len(masses))
deltapperrs_fit = np.zeros(len(masses))

power = 0 #1. / 3.

f1, ax1 = plt.subplots(figsize=(10,8), dpi=100)
f2, ax2 = plt.subplots(figsize=(10,8), dpi=100)

f3, ax3 = plt.subplots(figsize=(10,8), dpi=100)

ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xscale('log')

ax1a = ax1.twinx()
ax2a = ax2.twinx()
ax1a.set_yscale('log')
ax2a.set_yscale('log')

bead_area = 4. * np.pi * (2.5e-6)**2  # m^2
force_to_pressure = (1.0e-2 / (0.5 * bead_area)) * 1e-15  # converts fN on bead -> mbar of deltaP felt

beads = pressure_dir.keys()
#print pressure_dir

#for i, bead in enumerate(beads):
    #if 'HERA' in bead:
    #    del beads[i]

for i, bead in enumerate(beads):
    if 'HERA' in bead:
        continue

    fig_title = 'Force at 20 um vs. Pressure'
    gases = pressure_dir[bead].keys()
    style = iterations[i]

    gases1 = []
    for gas in gases:
        print gas
        #if 'He' not in gas:
        #    if 'Ar' not in gas:
        #        continue
        if '3' in gas:
            continue
        if '2' not in gas:
            gases1.append(gas)
        else:
            if 'gases2' not in locals():
                gases2 = []
            gases2.append(gas)

    gases1.sort()
    if 'gases2' in locals():
        gases2.sort()
        gases = gases1[:] + gases2[:]
    else:
        gases = gases1

    for gas in gases:
        if '2' in gas:
            style = iterations[2*i+1]
        else:
            style = iterations[2*i]

        color = colors[gas[:2]][0]
        mass = colors[gas[:2]][1]
        
        label = bead + ', ' + gas[:2]
        pressures = pressure_dir[bead][gas]['raw'].keys()
        pressures.sort()
        sigma_p = []

        rawforces = []
        rawerrs = []
        fitforces = []
        fiterrs = []

        for pressure in pressures:

            rawforces.append(pressure_dir[bead][gas]['raw'][pressure][0] / (mass)**power)
            rawerrs.append(pressure_dir[bead][gas]['raw'][pressure][1] / (mass)**power)
            fitforces.append(pressure_dir[bead][gas]['fit'][pressure][0] / (mass)**power)
            fiterrs.append(pressure_dir[bead][gas]['fit'][pressure][1] / (mass)**power)
            
            sigma_p.append(0.5 * (pressure_dir[bead][gas]['raw'][pressure][2] + \
                                   pressure_dir[bead][gas]['fit'][pressure][2]))

        pressures = np.array(pressures)
        rawforces = np.abs(np.array(rawforces))
        rawerrs = np.array(rawerrs)
        fitforces = np.abs(np.array(fitforces))
        fiterrs = np.array(fiterrs)

        deltap_raw = rawforces*1e15 * force_to_pressure 
        deltap_fit = fitforces*1e15 * force_to_pressure 

        ind = np.argmin(np.abs(mass - masses))
        deltapp_raw[ind] = np.mean(deltap_raw / pressures) * 100.
        deltapperrs_raw[ind] = np.std(deltap_raw / pressures) / len(pressures) * 100.
        deltapp_fit[ind] = np.mean(deltap_fit / pressures) * 100.
        deltapperrs_fit[ind] = np.std(deltap_fit / pressures) / len(pressures) * 100.

        ax1.errorbar(pressures, rawforces*1e15, xerr = sigma_p, yerr = rawerrs*1e15, \
                     fmt=style, label=label, color=color)
        ax1a.errorbar(pressures, rawforces*1e15 * force_to_pressure, \
                      xerr = sigma_p, yerr = rawerrs*1e15, \
                      fmt=style, label=label, color=color)

        ax2.errorbar(pressures, fitforces*1e15, xerr = sigma_p, yerr = fiterrs*1e15, \
                     fmt=style, label=label, color=color)
        ax2a.errorbar(pressures, fitforces*1e15 * force_to_pressure, \
                      xerr = sigma_p, yerr = fiterrs*1e15, \
                      fmt=style, label=label, color=color)

        if 'xmin' not in locals():
            xmin = np.min(pressures)
            xmax = np.max(pressures)
            yminr = 1e15 * np.min(rawforces)
            ymaxr = 1e15 * np.max(rawforces)
            yminf = 1e15 * np.min(fitforces)
            ymaxf = 1e15 * np.max(fitforces)

        if xmin > np.min(pressures):
            xmin = np.min(pressures)

        if xmax < np.max(pressures):
            xmax = np.max(pressures)

        if yminf > 1e15 * np.min(fitforces):
            yminf = 1e15 * np.min(fitforces)

        if ymaxf < 1e15 * np.max(fitforces):
            ymaxf = 1e15 * np.max(fitforces)

        if yminr > 1e15 * np.min(rawforces):
            yminr = 1e15 * np.min(rawforces)

        if ymaxr < 1e15 * np.max(rawforces):
            ymaxr = 1e15 * np.max(rawforces)
        

ax3.errorbar(masses, deltapp_raw, deltapperrs_raw, label='Raw', fmt='.-', ms=10)
ax3.errorbar(masses, deltapp_fit, deltapperrs_fit, label='Fit', fmt='.-', ms=10)
ax3.legend(loc=0, numpoints=2, ncol=2, fontsize=15)
ax3.set_xlabel('Mass of Residual Gas Atom [amu]')
ax3.set_ylabel('Differential Pressure [%]')

ax1.set_xlabel('Residual Gas Pressure [mbar]')
ax1.set_ylabel('Y-direction Force at 20 um [fN]')
ax1.set_ylim(0.667 * yminr, 1.5 * ymaxr)
ax1.set_xlim(0.667 * xmin, 1.5 * xmax)

ax2.set_xlabel('Residual Gas Pressure [mbar]')
ax2.set_ylabel('Y-direction Force at 20 um [fN]')
ax2.set_ylim(0.667 * yminf, 1.5 * ymaxf)
ax2.set_xlim(0.667 * xmin, 1.5 * xmax)


ax1a.set_ylabel('Differential Pressure [mbar]', rotation=90)
ax1a.set_ylim(0.667 * yminr * force_to_pressure, 1.5 * ymaxr * force_to_pressure)

ax2a.set_ylabel('Differential Pressure [mbar]', rotation=90)
ax2a.set_ylim(0.667 * yminf * force_to_pressure, 1.5 * ymaxf * force_to_pressure)

ax1.legend(loc=0, numpoints=1, ncol=2, fontsize=9)
ax1.grid(which='major', color='0.1')
ax1.grid(which='minor', color ='0.65')
f1.suptitle('Force vs. Pressure - Raw Data, HERMES-160808', fontsize=18)

ax2.legend(loc=0, numpoints=1, ncol=2, fontsize=9)
ax2.grid(which='major', color='0.1')
ax2.grid(which='minor', color ='0.65')
f2.suptitle('Force vs. Pressure - Fit to Data, HERMES-160808', fontsize=18)

plt.show()
        
