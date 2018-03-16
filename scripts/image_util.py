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
import re
import peakdetect as pdet
from scipy.optimize import curve_fit
import scipy.stats

#Functions for use in the class representing image data.

b, a = signal.butter(4, [.005, .5], btype = 'bandpass')

def marg(imarr, axis):
    '''Subtracts median off of image array and then margenalizes
       along specified axis.'''
    return np.sum(imarr-np.median(imarr), axis = axis)

def sem(sigs):
    '''standard error of weighted mean.'''
    return np.sqrt(1./np.sum(sigs**-2))

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
    ix = signal.filtfilt(b, a, gaussWeight(marged, edge0))
    return ix/np.sqrt(np.sum(ix**2))


def initYProcessing(imarr):
    '''margenaizes by summing over y, finds initia edge, and weights.'''
    marged = marg(imarr, 0) 
    iy = signal.filtfilt(b, a, marged)
    return iy/np.sqrt(np.sum(iy**2))

def getNanoStage(fname):
    '''Takes image filename as argument. gets nano positioning stage 
       dc settings by opening the .h5 file associated with the image file.
       Returns the median voltage times the stage calibration from V to um'''
    h5fname = os.path.splitext(fname)[0]
    df = bu.DataFile()
    df.load(h5fname)
    return np.median(df.cant_data, axis = -1)*configuration.stage_cal

def plotImages(Images):
    '''Plots an array of images'''
    n = int(np.ceil(np.sqrt(len(Images))))
    f, axarr = plt.subplots(n, n, sharex = True)
    i = 0
    for r in range(n):
        for c in range(n):
            axarr[r, c].imshow(Images[i].imarr)
            i += 1
            if i+1 > len(Images):
                break
        if i+1 > len(Images):
            break
    plt.show()

def picoSortFun(path):
    '''gets integer off of end of pico motor path for sorting'''
    return int(re.match('.*?([0-9]+)$', path).group(1))


def getPaths(path):
    '''returns list of pico motor paths inside parent directory'''
    paths = [x[0] for x in os.walk(path)][1:]#chop off 0th parent directory
    paths.sort(key = picoSortFun)
    return paths


def findMaxCorr(corr, make_plot = False, \
        lookahead = 10, delta = 0.05, rt = 0.8):
    '''picks the right peak out of the correlation.'''
    peaks = np.array(pdet.peakdetect(corr, \
            lookahead = lookahead, delta = delta)[0]) #[0] to get positive 
    inds = peaks[:, 0]
    values = peaks[:, 1]
    maxPeak = np.argmax(values)
    candidates = values>rt*values[maxPeak]
    #return middle candidate value if odd number of candidates
    ncandids = np.sum(candidates)
    if ncandids%2 and ncandids >1:
        ind = np.median(inds[candidates])
        #make_plot = True
    else:
        ind = inds[maxPeak]
    ind = int(ind)
    if make_plot:
        plt.plot(corr)
        plt.plot([ind], corr[ind], 'x')
        plt.show()
    return ind

def line(x, m, b):
    '''line function for fitting'''
    return m*x + b

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
                     invertY = True, peak_detect = False):
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
        if peak_detect:
            offset = cent - findMaxCorr(corr)
        else:
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
        
        #ability to treat axees differently
        if axis == 0:
            return np.array([offset, np.max(corr)])
        else:
            return np.array([-1*offset, np.max(corr)])


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
        self.indArr = self.ind_arr()
        self.shape = np.shape(self.indArr)

    def groupAxis(self, axis, thresh = 0.25):
        '''groups inds of images into rows or columns from nano 
           positioning stage measurements. Sorted into group when
           difference below thresh.'''
        pax = self.nanoPs[axis, :] #position array
        sortInds = np.argsort(pax) #sort index array
        inds = [] #initialize list to store lists of inds for each group
        rowCol = [] #initialize list for each group
        init_p = pax[sortInds[0]] 
        for i in sortInds:
            if pax[i] - init_p>thresh:
                inds.append(rowCol)
                rowCol = [i]
                init_p = pax[i]
            if i == sortInds[-1]:
                rowCol.append(i)
                inds.append(rowCol)
            else:
                rowCol.append(i)

        return inds

    def ind_arr(self, thresh = 0.25):
        '''returns a 2d nump array of image indicies indexed by image  
           row and column. If there are multiple images at the same grid
           location it returns the first.'''

        xinds = self.groupAxis(0, thresh = thresh)
        yinds = self.groupAxis(1, thresh = thresh)
        nx = len(xinds)
        ny = len(yinds)
        indArr = np.zeros((nx, ny), dtype = int)
        for i in range(nx):
            for j in range(ny):
                ind = list(set(xinds[i]).intersection(yinds[j]))
                indArr[i, j] = ind[0]

        return indArr


    def measureImage(self, image, makePlots = False, pltFit = False):
        '''Finds the location of an image in the image grid by fitting
            to the pixel shift from images that correlate most with 
            image.'''
        mx = lambda im: im.measureShift(image, 0, plotCorr = makePlots,\
                                        makePlot = makePlots)
        my =  lambda im: im.measureShift(image, 1, plotCorr = makePlots,\
                                        makePlot = makePlots)
        xShifts = np.array(map(mx, self.images))
        yShifts = np.array(map(my, self.images))
        cent_im = np.argmin((1-yShifts[:, 1])**2 + (1-xShifts[:, 1])**2)
        cent_ind = zip(*np.where(self.indArr == cent_im))[0]
        # determine edge cases 
        bxl = cent_ind[0]>0 
        bxh = cent_ind[0] < self.shape[0]-1
        byl = cent_ind[1]>0 
        byh = cent_ind[1] < self.shape[1]-1
        #do case away from edges
        if bxl and bxh and byl and byh:
            fitinds = np.ndarray.flatten(self.indArr[\
                    cent_ind[0]-1:cent_ind[0]+2,\
                    cent_ind[1]-1:cent_ind[1] +2])
            xfitShifts = xShifts[fitinds, 0]
            xnanoPs = self.nanoPs[0, fitinds]
            yfitShifts = yShifts[fitinds, 0]
            ynanoPs = self.nanoPs[1, fitinds]
            xpopt, xcov = curve_fit(line, xfitShifts, xnanoPs)
            ypopt, ycov = curve_fit(line, yfitShifts, ynanoPs)
            if pltFit:
                f, axarr = plt.subplots(1, 2, sharex = True, sharey = True)
                axarr[0].plot(xfitShifts, xnanoPs, 'o')
                axarr[0].plot(xfitShifts, line(xfitShifts, *xpopt))
                axarr[1].plot(yfitShifts, ynanoPs, 'o')
                axarr[1].plot(yfitShifts, line(yfitShifts, *ypopt))
                axarr[0].set_xlabel("x pixel shift")
                axarr[0].set_ylabel("x nano positiong stage [um]")
                axarr[1].set_xlabel("y pixel shift")
                axarr[1].set_ylabel("y nano positiong stage [um]")
                plt.show()
            return np.array([[xpopt[-1], np.sqrt(xcov[-1, -1])], \
                            [ypopt[-1], np.sqrt(ycov[-1, -1])]])
        else:
             return np.array([[np.nan, np.nan], \
                            [np.nan, np.nan]])
        
    def measureGrid(self, ImageGrid2, make_plot = False):
        '''uses self to measure every image in ImageGrid2. 
        From the differences in nano positioning stage at 
        the same image location determines the shift in the 
        nano positioning stage.'''
        #position of images in ImageGrid2 relative to self
        pos21 = []
        for im in ImageGrid2.images:
            pos21.append(self.measureImage(im))
	pos21 = np.array(pos21) #[image, x or y, value or sigma]
	#get images with measured positions
        valid = np.bitwise_not(np.isnan(pos21[:, 0, 0]))
	deltaXs = pos21[valid, 0, 0]\
                - np.transpose(ImageGrid2.nanoPs)[valid, 0]
        sigXs = pos21[valid, 0, 1]
	deltaYs = pos21[valid, 1, 0]\
		 - np.transpose(ImageGrid2.nanoPs)[valid, 1]
        sigYs = pos21[valid, 1, 1]
	xShift = np.average(deltaXs, weights = sigXs**-2)
        sigXshift = sem(sigXs)
	yShift = np.average(deltaYs, weights = sigYs**-2)
        sigYshift = sem(sigYs)

	if make_plot:
                plt.plot(pos21[valid, 0, 0], pos21[valid, 1, 0], '.')
                plt.show()
                ybins = np.arange(10, 40)
		plt.subplot(211), plt.hist(deltaXs)
		plt.subplot(212), plt.hist(deltaYs)
		plt.xlabel("shift [um]")
		plt.ylabel("Number")
		plt.show()
		
		plt.plot(self.nanoPs[0, :], self.nanoPs[1, :],'.',  \
			label = 'self grid')
		plt.plot(ImageGrid2.nanoPs[0, :] + xShift, \
			 ImageGrid2.nanoPs[1, :] + yShift, '.',\
			label = 'new grid shifted')	
		plt.plot(ImageGrid2.nanoPs[0, valid] + xShift, \
			 ImageGrid2.nanoPs[1, valid] + yShift, '.',\
			label = 'overlap')
		plt.legend()
		plt.xlabel('x position [um]')
       		plt.ylabel('y position [um]')
		plt.show()	
	
	return [[xShift, sigXshift], [yShift, sigYshift]]


def plotPicoPos(igs):
    '''takes array of image grids and plots the change in position'''
    pxs = np.zeros_like(igs)
    sigxs = np.zeros_like(igs)
    pys = np.zeros_like(igs)
    sigys = np.zeros_like(igs)

    for i, ig in enumerate(igs[1:]):
        shift = igs[i].measureGrid(ig)
        pxs[i+1] = shift[0][0] + pxs[i]
        pys[i+1] = shift[1][0] + pys[i]
        sigxs[i+1] = np.sqrt(np.sum(sigxs**2) + shift[0][1]**2)
        sigys[i+1] = np.sqrt(np.sum(sigys**2) + shift[1][1]**1)
    
    plt.errorbar(pxs, pys, xerr = sigxs*10., yerr = sigys*10., fmt = 'o-', label = '10 $\sigma$')
    plt.xlabel("x position [um]")
    plt.ylabel('y position [um]')
    plt.legend()
    plt.show()




