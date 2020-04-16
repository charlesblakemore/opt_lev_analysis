import os, re, time, sys, inspect, traceback
import numpy as np
import dill as pickle 

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.mlab as mlab

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy






class ECDF:
    '''Empirical cumulutive distribution function following
       Thomas Sargent and John Stachurski
       https://python.quantecon.org
    '''

    def __init__(self, samples):
        '''Load the samples in, don't do anything else
        '''
        self.samples = np.array(samples).flatten()
        self.nsamp = len(self.samples)


    def __call__(self, x):

        out = np.zeros(len(x))
        for ind, val in enumerate(x):
            out[ind] = np.sum(self.samples < val)
        return out / self.nsamp


    def build_interpolator(self, npts=100, limfacs=(2.0,2.0), smoothing=0.0):
        ### Find the min/max of the sample
        minval = np.min(self.samples)
        maxval = np.max(self.samples)

        ### Extend the limits of the interpolator beyond the limits of the sample
        ### to avoid edge effects (assumes stuff about the moments of the 
        ### underlying distribution, but I'm not a statistician)
        lower = limfacs[0]**(-1.0*np.sign(minval)) * minval
        upper = limfacs[1]**(1.0*np.sign(maxval)) * maxval

        ### Contruct the explicit ECDF
        x_arr = np.linspace(lower, upper, int(np.max(limfacs)*npts))
        y_arr = self(x_arr)

        ### Interpolate the ECDF and add some smoothing if desired
        func = interp.UnivariateSpline(x_arr, y_arr)
        func.set_smoothing_factor(smoothing)

        return func


    def PDF(self, npts=100, limfacs=(2.0,2.0), smoothing=0.0):
        interp_func = self.build_interpolator(npts=npts, limfacs=limfacs, \
                                                smoothing=smoothing)
        return interp_func.derivative(n=1)
        





class ECDF2:
    '''Empirical cumulutive distribution attemp to vectorize
       initial testing seemed to imply it's slower than the loop
    '''

    def __init__(self, samples):
        '''Load the samples in, don't do anything else
        '''
        self.samples = np.array(samples).flatten()
        self.nsamp = len(self.samples)

    def __call__(self, x):

        sample_arr = np.tile(self.samples, (len(x), 1))
        x_arr = np.reshape(x, (len(x),1))

        out_arr = np.zeros((len(x), self.nsamp))
        np.less(sample_arr, x_arr, out=out_arr)

        return np.sum(out_arr, axis=-1) / self.nsamp