import numpy
from pylab import *
from scipy.interpolate import interp1d

d1,e1,ee1,f1,ef1=numpy.loadtxt("full.txt",unpack=True)
f1=-f1*31.6e-15
inds=argsort(d1)
d1=d1[inds]
f1=f1[inds]

d2,e2,ee2,f2,ef2=numpy.loadtxt("PEC.txt",unpack=True)
f2=-f2*31.6e-15
inds=argsort(d2)
d2=d2[inds]
f2=f2[inds]

d3,e3,ee3,f3,ef3=numpy.loadtxt("temp.txt",unpack=True)
f3=-f3*31.6e-15
inds=argsort(d3)
d3=d3[inds]
f3=f3[inds]

datafile="../../Mathematica/calculated_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(datafile,unpack=True)
dist=dist*1e6

PFA_datafile="../../Mathematica/calculated_pfa_vals.tsv"
dist2,fpfa2,fnaive2,fright2,ftemp2=numpy.loadtxt(PFA_datafile,unpack=True)
dist2=dist2*1e6

plot(d2,f2,label="PEC")
plot(d1,f1,label="SiO2/Au")
plot(d3,f3,label="PEC T=300")

plot(dist,fpfa,label="PFA",linestyle='dashed')
plot(dist,fright,label="SiO2/Au",linestyle='dashed')
plot(dist,ftemp,label="SiO2/Au T=300",linestyle='dashed')
xscale('log')
yscale('log')
xlabel('Distance (microns)')
ylabel('Force (N)')
title('Analytical (Dashed) v Numerical (Solid) Calculations')
legend()
show()
#savefig('analytic_v_numerical')

clf()
iPFA = interp1d(dist,fpfa)
rPFA=iPFA(d2)/f2
iPFA2 = interp1d(dist2,fpfa2)
rPFA2=iPFA2(d2)/f2
plot(d2,rPFA,label="Gradient Expansion PFA")
plot(d2,rPFA2,label="Normal PFA")
xscale('log')
yscale('log')
xlabel('Distance (Microns)')
ylabel('(PFA/BEM) Force Ratio')
title('Comparion between Calculations, grid=1 micron')
legend()
savefig("pfa_v_pec.png")

clf()
tag,xi,e,f=numpy.loadtxt("mesh_byXi.txt",unpack=True)
f=-f*31.6e-15
xis=numpy.unique(xi)
for ix in range(0,len(xis)):
    inds = numpy.where(xi == xis[ix])
    xplot=tag[inds]
    yplot=f[inds]
    plot(xplot,yplot,label=str(xis[ix]))
    
xscale('log')
yscale('log')
xlim(0.5,5.0)
legend(title='Int. Frequency',loc='lower left')
xlabel('Mesh length scale (microns)')
ylabel('Force Integrand Value')
title('Convergence Tests @ 1 micron')
savefig('convergence.png')
