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
import glob

#Functions for use in the class representing image data.

b, a = signal.butter(4, [.02, .5], btype = 'bandpass')

def marg(imarr, axis):
    '''Subtracts median off of image array and then margenalizes
       along specified axis.'''
    return np.sum(imarr-np.median(imarr), axis = axis)

def initEdge(arr, thresh = 0.5):
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
    return signal.filtfilt(b, a, gaussWeight(marged, edge0))


def initYProcessing(imarr):
    '''margenaizes by summing over y, finds initia edge, and weights.'''
    marged = marg(imarr, 0) 
    return signal.filtfilt(b, a, marged)

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
                               initYProcessing(self.imarr)])
        self.nanoPos = getNanoStage(self.fname)        


    def measureShift(self, Image2, axis, plotCorr = True, makePlot = True,\
                     invertY = True):
        '''measures shift between self and Image2 from the shift in 
           the maximum of the correlation and auto correlation of images
           margenalized over opposite axis. If shift is positive then Image 2
           is to the right of self.'''

        cent = np.max([len(self.margs[axis]), len(Image2.margs[axis])])-1     
        corr = signal.correlate(self.margs[axis], Image2.margs[axis])
        #acorr = signal.correlate(self.margs[axis], self.margs[axis])
        if plotCorr:
            plt.plot(np.arange(len(corr)) - cent, corr, label = 'correlation')
            plt.xlabel('pixel shift')
            plt.ylabel('correlation')
            plt.legend()
            plt.show()
        #only shifts discrete pixels. need to fix
        #if offset is >0 then image 2 is ahead of self
        offset = cent - np.argmax(corr)
        if makePlot:
            if offset>=0:
                #advance self to right
                plt.plot(np.concatenate((np.zeros(offset), self.margs[axis])),\
                         '.', label = 'self shifted right ' + str(offset))
            else:
                # move Image2 right
                plt.plot(np.concatenate((np.zeros(-1*offset),\
                Image2.margs[axis])), '.', label = 'image2 shifted right ' \
                + str(-1*offset))
            plt.plot(self.margs[axis], label = 'self')
            plt.plot(Image2.margs[axis], label = 'image 2')
            plt.legend()
            plt.show()
        
        if axis == 0:
            return offset
        else:
            return -1*offset


class ImageGrid:
    '''Class for storing a collection of Image objects. 
     Contains a method that can determine the location of an arbitrarily 
     shifted image within the grid.'''


    def __init__(self, path):
        '''loads images from path into list of image objects'''
        imArr = []
        imFnames = glob.glob(path + '/*.h5.npy')
        for fname in imFnames:
            imArr.append(Image(fname))
        self.fnames = imFnames
        self.images = imArr
        posEr = lambda image: image.nanoPos 
        self.nanoPs = np.transpose(np.array(map(posEr, self.images)))



    def measureImage(self, image, makePlots = False, rmax = 10):
        '''Finds all of the pixel shifts between Image and each 
           Image in self.'''
        mx = lambda im: im.measureShift(image, 0, plotCorr = makePlots,\
                                        makePlot = makePlots)
        my =  lambda im: im.measureShift(image, 1, plotCorr = makePlots,\
                                        makePlot = makePlots)
        xShifts = map(mx, self.images)
        yShifts = map(my, self.images)
        prs = np.sqrt(xShifts**2 + yShifts**2)
        neighbors = np.arange(len(self.images))[prs < rmax]
        nn = np.argmin(prs)










 
           





