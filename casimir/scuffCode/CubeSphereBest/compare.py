import numpy
from pylab import *
from scipy.interpolate import interp1d

d1t,g1t,e1t,ee1t,f1t,ef1t,st=numpy.loadtxt("PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1t=-f1t*31.6e-15
inds=argsort(d1t)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]
st=st[inds]
inds=numpy.where(st == 0)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]

d1,g1,e1,ee1,f1,ef1,s=numpy.loadtxt("PEC_combined_results.txt",unpack=True,skiprows=1)
f1=-f1*31.6e-15
inds=argsort(d1)
d1=d1[inds]
f1=f1[inds]
g1=g1[inds]
s=s[inds]
inds=numpy.where(s == 0)
d1=d1[inds]
f1=f1[inds]
g1=g1[inds]

d1ts,g1ts,e1ts,ee1ts,f1ts,ef1ts,sts=numpy.loadtxt("../CubeSphere/PEC_combined_results_temp.txt",unpack=True,skiprows=1)
f1ts=-f1ts*31.6e-15
inds=argsort(d1ts)
d1ts=d1ts[inds]
f1ts=f1ts[inds]
g1ts=g1ts[inds]
sts=sts[inds]
inds=numpy.where(sts == 0)
d1ts=d1ts[inds]
f1ts=f1ts[inds]
g1ts=g1ts[inds]

d2t,g2t,e2t,ee2t,f2t,ef2t,s2t=numpy.loadtxt("combined_results_temp.txt",unpack=True,skiprows=1)
f2t=-f2t*31.6e-15
inds=argsort(d2t)
d2t=d2t[inds]
f2t=f2t[inds]
g2t=g2t[inds]
s2t=s2t[inds]
inds=numpy.where(s2t == 0)
d2t=d2t[inds]
f2t=f2t[inds]
g2t=g2t[inds]

d2,g2,e2,ee2,f2,ef2,s2=numpy.loadtxt("combined_results.txt",unpack=True,skiprows=1)
f2=-f2*31.6e-15
inds=argsort(d2)
d2=d2[inds]
f2=f2[inds]
g2=g2[inds]
s2=s2[inds]
inds=numpy.where(s2 == 0)
d2=d2[inds]
f2=f2[inds]
g2=g2[inds]

d2ts,g2ts,e2ts,ee2ts,f2ts,ef2ts,sts2=numpy.loadtxt("../CubeSphere/combined_results_temp.txt",unpack=True,skiprows=1)
f2ts=-f2ts*31.6e-15
inds=argsort(d2ts)
d2ts=d2ts[inds]
f2ts=f2ts[inds]
g2ts=g2ts[inds]
sts2=sts2[inds]
inds=numpy.where(sts2 == 0)
d2ts=d2ts[inds]
f2ts=f2ts[inds]
g2ts=g2ts[inds]

datafile="../../Mathematica/calculated_vals.tsv"
PFA_datafile="../../Mathematica/calculated_pfa_vals.tsv"
EXP_datafile="../../Mathematica/calculated_exp_vals.tsv"
dist,fpfa,fnaive,fright,ftemp=numpy.loadtxt(PFA_datafile,unpack=True)
dist2,fexp,fexptemp,fexpfin,fexpfintemp=numpy.loadtxt(EXP_datafile,unpack=True)
fexp=fexp*1e-18
fexptemp=fexptemp*1e-18
fexpfin=fexpfin*1e-18
fexpfintemp=fexpfintemp*1e-18
dist=dist*1e6

figure(figsize=(12,8))

gst=numpy.min(g1t)
inds = numpy.where(g1t == gst)
xnew=np.arange(numpy.min(d1t[inds]), 30, 0.1)
s = interp1d(d1t[inds],log(f1t[inds]),kind='cubic')
plot(xnew,exp(s(xnew)),'--',color="red")
scatter(d1t[inds],f1t[inds],marker='o',label="P 300K, g="+str(gst),color="red")

gs=numpy.min(g1)
inds = numpy.where(g1 == gs)
xnew=np.arange(numpy.min(d1[inds]), 30, 0.1)
s = interp1d(d1[inds],log(f1[inds]),kind='cubic')
plot(xnew,exp(s(xnew)),'--',color="blue")
scatter(d1[inds],f1[inds],marker='o',label="P 0K, g="+str(gs),color="blue")
inds = numpy.where(g1 == 0.5)
#plot(d1[inds],f1[inds],'-d',label="P 0K, g="+str(0.4),color="blue")

gs=numpy.min(g2t)
inds = numpy.where(g2t == gs)
xnew=np.arange(numpy.min(d2t[inds]), 30, 0.1)
s = interp1d(d2t[inds],log(f2t[inds]),kind='cubic')
plot(xnew,exp(s(xnew)),'--',color="green")
scatter(d2t[inds],f2t[inds],marker='o',label="F 300K, g="+str(gs),color="green")
inds = numpy.where(g2t == 0.5)
scatter(d2t[inds],f2t[inds],marker='d',label="F 300K, g="+str(0.5),color="green")

gs=numpy.min(g2)
inds = numpy.where(g2 == gs)
scatter(d2[inds],f2[inds],marker='o',label="F 0K, g="+str(gs),color="black")
inds = numpy.where(g2 == 0.5)
xnew=np.arange(numpy.min(d2[inds]), 30, 0.1)
s = interp1d(d2[inds],log(f2[inds]),kind='cubic')
plot(xnew,exp(s(xnew)),'--',color="black")
scatter(d2[inds],f2[inds],marker='d',label="F 0K, g="+str(0.5),color="black")

gst=numpy.min(g1ts)
inds = numpy.where(g1ts == gst)
#plot(d1ts[inds],f1ts[inds],'-.',label="P 300K, cube, g="+str(gst),color="red")

gs=numpy.min(g2ts)
inds = numpy.where(g2ts == gs)
#plot(d2ts[inds],f2ts[inds],'-.',label="F 300K, cube, g="+str(gs),color="green")

plot([1,30],[1e-15,1e-15],':',color="black")
plot([1,30],[1e-16,1e-16],':',color="black")
plot([1,30],[1e-17,1e-17],':',color="black")
plot([1,30],[1e-18,1e-18],':',color="black")
plot([1,30],[1e-19,1e-19],':',color="black")
plot([1,30],[1e-20,1e-20],':',color="black")
plot([1,30],[1e-21,1e-21],':',color="black")

#plot(dist,fpfa,label="PFA T=0",linestyle='--',color="blue")
plot(dist2,fexp,label="Corrected PFA T=0",linestyle='-',color="blue")
plot(dist2,fexptemp,label="Corrected PFA T=300",linestyle='-',color="red")
plot(dist2,fexpfin,label="Si/Au T=0",linestyle='-',color="black")
plot(dist2,fexpfintemp,label="Si/Au T=300",linestyle='-',color="green")
#plot(dist,fright,label="SiO2/Au T=0",linestyle='--',color="black")
#plot(dist,ftemp,label="SiO2/Au T=300",linestyle='--',color="green")
xlim(1,30)
#xscale('log')
yscale('log')
ylim(1e-21,1e-14)
xlabel('Distance (microns)')
ylabel('Force (N)')
title('Analytical v Numerical Calculations')
legend(loc="upper right",ncol=3)
savefig('analytic_v_numerical_best')
#show()

clf()
iPFA = interp1d(dist2,fexptemp)
gs=numpy.min(g1t)
inds = numpy.where(g1t == gs)
rPFA=f1t[inds]/iPFA(d1t[inds])
plot(d1t[inds],rPFA,label="PEC/PFA, g="+str(gs))
iPFA = interp1d(dist2,fexp)
gs=numpy.min(g1)
inds = numpy.where(g1 == gs)
rPFA=f1[inds]/iPFA(d1[inds])
plot(d1[inds],rPFA,label="PEC/PFA 0K, g="+str(gs))
iPFA = interp1d(dist2,fexpfintemp)
gs=numpy.min(g2t)
inds = numpy.where(g2t == gs)
rPFA=f2t[inds]/iPFA(d2t[inds])
rPFA=rPFA/rPFA[0]
plot(d2t[inds],rPFA,label="FEC/PFA, g="+str(gs))
iPFA = interp1d(dist2,fexpfin)
gs=numpy.min(g2)
inds = numpy.where(g2 == gs)
rPFA=f2[inds]/iPFA(d2[inds])
rPFA=rPFA/rPFA[0]
plot(d2[inds],rPFA,label="FEC/PFA 0K, g="+str(gs))
iPFA = interp1d(dist2,fexpfintemp)
xlim(1,30)
xlabel('Distance/Radius')
ylabel('(BEM/PFA) Force Ratio')
title('Comparion between Calculations, grid=1 micron')
legend()
savefig("geometry_corrections.png")

#data points computed (through similar method) for correction due to aspect ratio L/R from PFA (Canaguier-Durand 2012)
cdx=[0,0.1,.2,0.4,0.6,0.8,1]
cdy=[1.0,.98,.95,.86,.78,.72,.68]

clf()
iPFA = interp1d(dist,fpfa)
gs=numpy.unique(g1t)
for i in range(0,len(gs)):
    inds = numpy.where(g1t == gs[i])
    rPFA=f1t[inds]/iPFA(d1t[inds])
    plot(d1t[inds]/2.5,rPFA,label="PEC/PFA, g="+str(gs[i]))
plot(cdx,cdy,label="Canaguieier-Durand",linestyle=':',color="black")
#xscale('log')
xlim(0,3)
xlabel('Distance/Radius')
ylabel('(PFA/BEM) Force Ratio')
title('Comparion between Calculations, grid=1 micron')
legend()
#show()
savefig("pfa_v_pec_best.png")

clf()
inds=argsort(g1t)
d1t=d1t[inds]
f1t=f1t[inds]
g1t=g1t[inds]
ds=numpy.unique(d1t)
for i in range(0,len(ds)):
    inds=numpy.where(d1t == ds[i])
    plot(g1t[inds],f1t[inds]/f1t[inds[0][0]],'--',label=str(ds[i]),alpha=.9)
plot([0.1,1.2],[1,1],linestyle=':',color='black')
ylim(0.9,1.05)
xlim(0.3,0.5)
xlabel('Grid Scale Length')
ylabel('Force/Force(smallest gridding)')
title("Convergence in Grid Spacing")
legend(loc='lower left',title="Separation",ncol=2)
savefig("pfa_convergence_zoom_best.png")

clf()
id1ts = interp1d(d1ts,f1ts)
gpts=where(g2ts == numpy.min(g2ts))
scatter(d2ts[gpts],f2ts[gpts]/id1ts(d2ts[gpts]))
savefig('conductivity_correction.png')

