import h5py, os, sys, re, glob, time, sys, fnmatch, inspect, subprocess, math, xmltodict
import numpy as np
import datetime as dt
import dill as pickle 

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.mlab as mlab

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.stats as stats
import scipy.constants as constants
import scipy

import configuration
import transfer_func_util as tf
from bead_util_funcs import find_all_fnames

import warnings



calib_path = '/data/old_trap_processed/calibrations/'



kb = constants.Boltzmann
Troom = 297 # Kelvins

# From 2019 mass and density paper
rhobead_arr = np.array([1.5499, 1.5515, 1.5624])
rhobead_sterr_arr = rhobead_arr * np.array([0.8/84.0, 1.1/83.9, 0.2/85.5])
rhobead_syserr_arr = \
        rhobead_arr * np.sqrt(np.array([1.5/84.0, 1.5/83.9, 1.5/85.5])**2 + \
                              9 * np.array([0.038/2.348, 0.037/2.345, 0.038/2.355])**2)

rhobead = {}
rhobead['val'] = 1e3 * np.sum(rhobead_arr * (1.0 / (rhobead_sterr_arr**2 + rhobead_syserr_arr**2))) / \
                    np.sum( 1.0 / (rhobead_sterr_arr**2 + rhobead_syserr_arr**2) )  # kg/m^3
rhobead['sterr'] = 1e3 * np.sqrt( 1.0 / np.sum(1.0 / rhobead_sterr_arr) ) 
rhobead['syserr'] = 1e3 * np.mean(rhobead_syserr_arr)  # Can't average away systematics





def get_mbead(date, verbose=False):
    '''Scrapes standard directory for measured masses with dates matching
       the input string. Computes the combined statistical and systematic
       uncertainties

           INPUTS: date, string in the format "YYYYMMDD" for the bead of interest
                   verbose, print some shit

           OUTPUTS:     Dictionary with keys:
                    val, the average mass (in kg)) from all measurements
                    sterr, the combined statistical uncertainty
                    syserr, the mean of the individual systematic uncertainties
    '''
    dirname = os.path.join(calib_path, 'masses/')
    mass_filenames, lengths = find_all_fnames(dirname, ext='.mass', verbose=False)

    if verbose:
        print('Finding files in: ', dirname)
    real_mass_filenames = []
    for filename in mass_filenames:
        if date not in filename:
            continue
        if verbose:
            print('    ', filename)
        real_mass_filenames.append(filename)

    masses = []
    sterrs = []
    syserrs = []
    for filename in real_mass_filenames:
        mass_arr = np.load(filename)
        masses.append(mass_arr[0])
        sterrs.append(mass_arr[1])
        syserrs.append(mass_arr[2])
    masses = np.array(masses)
    sterrs = np.array(sterrs)
    syserrs = np.array(syserrs)

    # Compute the standard, weighted arithmetic mean on all datapoints,
    # as well as combine statistical and systematic uncertainties independently
    mass = np.sum(masses * (1.0 / (sterrs**2 + syserrs**2))) / \
                np.sum( 1.0 / (sterrs**2 + syserrs**2) )
    sterr = np.sqrt( 1.0 / np.sum(1.0 / sterrs**2 ) )
    #syserr = np.sqrt( 1.0 / np.sum(1.0 / syserrs**2 ) )
    syserr = np.mean(syserrs)

    if verbose:
        print()
        print('                       Mass [kg] : {:0.4g}'.format(mass))
        print('Relative statistical uncertainty : {:0.4g}'.format(sterr/mass))
        print(' Relative systematic uncertainty : {:0.4g}'.format(syserr/mass))
        print()

    return {'val': mass, 'sterr': sterr, 'syserr': syserr}


def get_rbead(mbead={}, date='', rhobead=rhobead, verbose=False):
    '''Computes the bead radius from the given mass and an assumed density.
       Loads the mass if a date is provided instead of a mass dictionary

           INPUTS: mbead, dictionary output from get_mbead()
                   date, string in the format "YYYYMMDD" for the bead of interest
                   rhobead, density dictionary (default: hardcoded above)
                   verbose, print some shit (default: False)

           OUTPUTS:    Dictionary with keys:
                    val, the the computed radius (in m) from the given mass
                            and the density found in 2019 mass paper
                    sterr, the combined statistical uncertainty
                    syserr, the mean of the individual systematic uncertainties
    '''
    if not len(list(mbead.keys())):
        if not len(date):
            print('No input mass or date given. What did you expect to happen?')
            return
        try:
            mbead = get_mbead(date, verbose=verbose)
        except:
            print("Couldn't load mass files")
            sys.exit()

    rbead = {}
    rbead['val'] = ( (mbead['val'] / rhobead['val']) / ((4.0/3.0)*np.pi) )**(1.0/3.0)
    rbead['sterr'] = rbead['val'] * np.sqrt( ((1.0/3.0)*(mbead['sterr']/mbead['val']))**2 + \
                                ((1.0/3.0)*(rhobead['sterr']/rhobead['val']))**2 )
    rbead['syserr'] = rbead['val'] * np.sqrt( ((1.0/3.0)*(mbead['syserr']/mbead['val']))**2 + \
                                ((1.0/3.0)*(rhobead['syserr']/rhobead['val']))**2 )

    if verbose:
        print()
        print('                      Radius [m] : {:0.4g}'.format(rbead['val']))
        print('Relative statistical uncertainty : {:0.4g}'.format(rbead['sterr']/rbead['val']))
        print(' Relative systematic uncertainty : {:0.4g}'.format(rbead['syserr']/rbead['val']))
        print()

    return rbead


def get_Ibead(mbead={}, date='', rhobead=rhobead, verbose=False):
    '''Computes the bead moment of inertia from the given mass and an assumed density.
       Loads the mass if a date is provided instead of a mass dictionary

           INPUTS: mbead, dictionary output from get_mbead()
                   date, string in the format "YYYYMMDD" for the bead of interest
                   rhobead, density dictionary (default: hardcoded above)
                   verbose, print some shit (default: False)

           OUTPUTS:    Dictionary with keys:
                    val, the computed moment (in kg m^2) from the given mass
                            and the density found in 2019 mass paper
                    sterr, the combined statistical uncertainty
                    syserr, the mean of the individual systematic uncertainties
    '''

    if not len(list(mbead.keys())):
        if not len(date):
            print('No input mass or date given. What did you expect to happen?')
            return
        try:
            mbead = get_mbead(date, verbose=verbose)
        except:
            print("Couldn't load mass files")
            sys.exit()

    Ibead = {}
    Ibead['val'] = 0.4 * (3.0 / (4.0 * np.pi))**(2.0/3.0) * \
                    mbead['val']**(5.0/3.0) * rhobead['val']**(-2.0/3.0)
    Ibead['sterr'] = Ibead['val'] * np.sqrt( ((5.0/3.0)*(mbead['sterr']/mbead['val']))**2 + \
                                   ((2.0/3.0)*(rhobead['sterr']/rhobead['val']))**2 )
    Ibead['syserr'] = Ibead['val'] * np.sqrt( ((5.0/3.0)*(mbead['syserr']/mbead['val']))**2 + \
                                    ((2.0/3.0)*(rhobead['syserr']/rhobead['val']))**2 )

    if verbose:
        print()
        print('      Moment of inertia [kg m^2] : {:0.4g}'.format(Ibead['val']))
        print('Relative statistical uncertainty : {:0.4g}'.format(Ibead['sterr']/Ibead['val']))
        print(' Relative systematic uncertainty : {:0.4g}'.format(Ibead['syserr']/Ibead['val']))
        print()

    return Ibead



def get_kappa(mbead={}, date='', T=Troom, rhobead=rhobead, verbose = False):
    '''Computes the bead kappa from the given mass, temperature, and an assumed 
       density. This is the geometric factor defining how the bead experiences
       torsional drag from surrounding gas. Loads the mass if a date is provided 
       instead of a mass dictionary

           INPUTS: mbead, dictionary output from get_mbead()
                   date, string in the format "YYYYMMDD" for the bead of interest
                   T, ambient temperature of rotor
                   rhobead, density dictionary (default: hardcoded above)
                   verbose, print some shit (default: False)

           OUTPUTS:    Dictionary with keys:
                    val, the computed kappa (in J^1/2 m^-4) f
                    sterr, the combined statistical uncertainty
                    syserr, the mean of the individual systematic uncertainties
    '''

    if not len(list(mbead.keys())):
        if not len(date):
            print('No input mass or date given. What did you expect to happen?')
            return
        try:
            mbead = get_mbead(date, verbose=verbose)
        except:
            print("Couldn't load mass files")

    kappa = {}
    kappa['val'] = ( (4.0 * np.pi * rhobead['val']) / (3.0 * mbead['val']) )**(4.0/3.0) * \
                np.sqrt( (9.0 * kb * T) / (32 * np.pi) )
    kappa['sterr'] = kappa['val'] * np.sqrt( ((4.0 / 3.0) * (mbead['sterr'] / mbead['val']))**2 + \
                                   ((4.0 / 3.0) * (rhobead['sterr'] / rhobead['val']))**2 )
    kappa['syserr'] = kappa['val'] * np.sqrt( ((4.0 / 3.0) * (mbead['syserr'] / mbead['val']))**2 + \
                                    ((4.0 / 3.0) * (rhobead['syserr'] / rhobead['val']))**2 )

    if verbose:
        print() 
        print('Torsional drag kappa value [J^1/2 m^-4] : {:0.4g}'\
                            .format(kappa['val']))

        print('       Relative statistical uncertainty : {:0.4g}'\
                            .format(kappa['sterr']/kappa['val']))
        print('                   rhobead contribution : {:0.4g}'\
                            .format(rhobead['sterr']/rhobead['val']))
        print('                     mbead contribution : {:0.4g}'\
                            .format(mbead['sterr']/mbead['val']))

        print('        Relative systematic uncertainty : {:0.4g}'\
                            .format(kappa['syserr']/kappa['val']))
        print('                   rhobead contribution : {:0.4g}'\
                            .format(rhobead['syserr']/rhobead['val']))
        print('                     mbead contribution : {:0.4g}'\
                            .format(mbead['syserr']/mbead['val']))
        print()

    return kappa