import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob

path = '/data/20171114/im_grid'

fs = glob.glob(path + '/*.h5.npy')

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

def measureXShift(imarr1, imarr2, makePlot = False):
    '''measures x shift between imarr1 and imarr2 from the shift in 
       the maximum of the correlation and auto correlation of images
       margenalized over the y axis.'''
    x1 = initXProcessing(imarr1)
    x2 = initXProcessing(imarr2)
    corr = signal.correlate(x1, x2)
    acorr = signal.correlate(x1, x1)
    #only shifts discrete pixels. need to fix
    offset = np.argmax(acorr) - np.argmax(corr)
    if makePlot:
        plt.plot(x2, label = 'reference')
        if offset>=0:
            plt.plot(np.concatenate((np.zeros(offset), x1)),\
                     label = 'measured')
        else:
             plt.plot(np.concatenate((x1, np.zeros(-1*offset))),\
                     label = 'measured')
           
        plt.legend()
        plt.show() 
    
    return offset


def measureYShift(imarr1, imarr2, makePlot = False):
    '''measures y shift between imarr1 and imarr2 from the shift
    in the maximum of the correlation and the autocorrelation of images 
    margenalized over the x axis.'''
    y1 = marg(imarr1, 0)
    y2 = marg(imarr2, 0)
    #may need to improve by some kind of weighting
    acorr = signal.correlate(y1, y1)
    corr = signal.correlate(y1, y2)
    #only shifts discrete pixels. Need to fix
    offset = np.argmax(acorr) - np.argmax(corr) 
    if makePlot:
        plt.plot(y2, label = 'reference')
        if offset>=0:
            plt.plot(np.concatenate((np.zeros(offset), y1)),\
                     label = 'measured')
        else:
            plt.plot(np.concatenate((y1, np.zeros(-1*offset))),\
                     label = 'measured')
        
        plt.legend()
        plt.show()

    return offset

