import numpy
from pylab import *
from scipy.interpolate import interp1d

d1,g1,a1,t1,e1,ee1,f1,ef1,s=numpy.loadtxt("PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1=-f1*31.6e-15
inds=argsort(d1)
d1=d1[inds]
f1=f1[inds]
a1=a1[inds]
g1=g1[inds]
s=s[inds]
inds=numpy.where(s == 0)
d1=d1[inds]
f1=f1[inds]
a1=a1[inds]
g1=g1[inds]

d2,g2,a2,t2,e2,ee2,f2,ef2,s2=numpy.loadtxt("combined_results_temp.txt",unpack=True,skiprows=1)
f2=-f2*31.6e-15
inds=argsort(d2)
d2=d2[inds]
f2=f2[inds]
a2=a2[inds]
g2=g2[inds]
s2=s2[inds]
inds=numpy.where(s2 == 0)
d2=d2[inds]
f2=f2[inds]
a2=a2[inds]
g2=g2[inds]

figure(figsize=(12,8))
#gs=numpy.unique(g1)

#for j in range(0,len(gs)):
gs=0.3
inds = numpy.where(g1 == gs)
xd1=d1[inds]
yf1=f1[inds]
asp=a1[inds]
asps=numpy.unique(asp)
for i in range(0,len(asps)):
    gpts=numpy.where(asps[i] == asp)
    plot(xd1[gpts],yf1[gpts],'-o',label="PEC, grid="+str(gs)+" asp="+str(asps[i]))

#gs=numpy.min(g2)
#inds = numpy.where(g2 == gs)
#plot(d2[inds],f2[inds],'--',label="FEC, grid="+str(gs),color="green")
xscale('log')
yscale('log')
xlabel('Distance (microns)')
ylabel('Force (N)')
xlim(10,30)
title('Numerical Calculations, Aspect Ratio')
legend(loc="lower left",ncol=2)
savefig('force_v_aspect')
#show()

clf()
gs=0.3
#for j in range(0,len(gs)):
inds = numpy.where(g1 == gs)
xd1=d1[inds]
yf1=f1[inds]
asp=a1[inds]
lens=numpy.unique(xd1)
for i in range(0,len(lens)):
    gpts=numpy.where(lens[i] == xd1)
    x=asp[gpts]
    y=yf1[gpts]/numpy.min(yf1[gpts])
    sinds=numpy.argsort(x)
    plot(x[sinds],y[sinds],'-o',label="g="+str(gs)+" l="+str(lens[i]))

xlabel('Aspect Ratio (W/H)')
ylabel('Force(Aspect)/Force(Aspect=2)')
title('Aspect v Force Numerical Calculations')
legend(loc="lower left",ncol=4)
ylim(0,4.0)
savefig('aspect_correction.png')
#show()
