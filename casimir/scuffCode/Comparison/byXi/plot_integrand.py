#!/usr/bin/env python

import numpy
import scipy.integrate
from pylab import *

datafile="../../../Mathematica/calculated_vals.tsv"

tag,x,e,f = numpy.loadtxt("data.txt",unpack=True)
tags=numpy.unique(tag)
flimit = numpy.zeros(len(tags))

for i in range(0,len(tags)):
    itag=tags[i]
    inds = numpy.where(tag == itag)
    xplot=x[inds]
    yplot=-f[inds]*31e-15
    isort=numpy.argsort(xplot)
    xplot = xplot[isort]
    yplot = yplot[isort]
    plot(xplot,yplot)
    flimit[i] = scipy.integrate.trapz(xplot,-yplot)
    

yscale('log')
xscale('log')
savefig('integrands.png')

clf()

dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(datafile,unpack=True)
dist=dist*1e6

plot(tags,flimit)
plot(dist,fpfa)
plot(dist,fright)
plot(dist,ftemp)
xscale('log')
yscale('log')
show()
