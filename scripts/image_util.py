################################################################################
#Set of utilities for determing the displacement
#of the picomotors from images.
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import bead_util as bu
import os
import configuration

#Functions for use in the class representing image data.

def marg(imarr, axis):
    '''Subtracts median off of image array and then margenalizes
       along specified axis.'''
    return np.sum(imarr-np.median(imarr), axis = axis)

def initEdge(arr, thresh = 0.2):
    '''Makes an initial guess at the x position by finding the first 
       element above threshold specified relative to the maximum value.'''
    threshValue = np.max(arr)*thresh
    return next(x[0] for x in enumerate(arr) if x[1] > threshValue)

def gaussWeight(arr, mu, sig = 50):
    '''multiplies arr by gaussian weight centered at mu with width sig'''
    weights = np.exp(-1.*(np.arange(len(arr))-mu)**2/(2.*sig**2))
    return arr*weights

def initXProcessing(imarr):
    '''margenaizes by summing over y, finds initia edge, and weights.'''
    marged = marg(imarr, 1) 
    edge0 = initEdge(marged)
    return gaussWeight(marged, edge0)

def getNanoStage(fname):
    '''Takes image filename as argument. gets nano positioning stage 
       dc settings by opening the .h5 file associated with the image file.
       Returns the median voltage times the stage calibration from V to um'''
    h5fname = os.path.splitext(fname)[0]
    df = bu.DataFile()
    df.load(h5fname)
    return np.median(df.cant_data, axis = -1)*configuration.stage_cal
    


class Image:
    'Class for storing and measuring images of attractors for metrology'

    def __init__(self, fname):
        '''initalizes class with filename'''
        self.fname = fname
        self.imarr = np.load(fname)
        self.margs = np.array([initXProcessing(self.imarr),\
                               marg(self.imarr, 0)]) #for now just margenalize y
        self.nanoPos = getNanoStage(self.fname)        


    def measureShift(self, Image2, axis, makePlot = False):
        '''measures shift between self and Image2 from the shift in 
           the maximum of the correlation and auto correlation of images
           margenalized over opposite axis. If shift is positive then Image 2
           is to the right of self.'''
        
        corr = signal.correlate(self.margs[axis], Image2.margs[axis])
        acorr = signal.correlate(self.margs[axis], self.margs[axis])
        #only shifts discrete pixels. need to fix
        offset = np.argmax(acorr) - np.argmax(corr)
        if makePlot:
            plt.plot(Image2.margs[axis], label = 'reference')
            if offset>=0:
                plt.plot(np.concatenate((np.zeros(offset), self.margs[axis])),\
                         label = 'measured')
            else:
                 plt.plot(np.concatenate((self.margs[axis], np.zeros(-1*offset))),\
                         label = 'measured')
               
            plt.legend()
            plt.show()
        
        return offset


class ImageGrid:
    'Class for storing a collection of Image objects. 
     Contains a method that can determine the location of an arbitrarily 
     shifted image within the grid.'

