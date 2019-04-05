
# In[2]:

import numpy as np
import matplotlib.pyplot as plt



# In[1]:

import matplotlib.mlab
from scipy.optimize import curve_fit
from scipy import interpolate


# In[3]:

import bead_util as bu


# In[4]:

def HOPSD(w, w0, g, A):
    return A/(w0**4*((2*w*g/w0)**2 + (1 - (w/w0)**2)**2))


# In[9]:

path = "/data/20180927/bead1/lower1"
files  = bu.find_all_fnames(path)


# Out[9]:

#     Finding files in: 
#     /data/20180927/bead1/lower1
#     Found 196 files...
# 

# In[6]:

phipbit = 0.0000833


# In[44]:

len(files)


# Out[44]:

#     20

# In[10]:

poptsx = []
pcovsx= []
poptsy = []
pcovsy = []
zsb = []
zfb = []
Asx = []
Asy = []
for f in files:
    df = bu.DataFile()
    df.load(f)
    psdx, freqs = matplotlib.mlab.psd(df.pos_data[0, :], Fs = df.fsamp, NFFT = 2**14)
    psdy, freqs = matplotlib.mlab.psd(df.pos_data[1, :], Fs = df.fsamp, NFFT = 2**14)
    Ncut = 10
    Nx = np.sum(psdx[Ncut:])
    Ny = np.sum(psdy[Ncut:])
    p0 = [2.*np.pi*300., .1, 1.E6]
    try:
        poptx, pcovx = curve_fit(HOPSD, freqs[Ncut:]*2.*np.pi, psdx[Ncut:]/Nx, p0)
        popty, pcovy = curve_fit(HOPSD, freqs[Ncut:]*2.*np.pi, psdy[Ncut:]/Ny, p0)
        #plt.loglog(freqs, psd, '.')
        #plt.plot(freqs, HOPSD(freqs*2.*np.pi, *popt)*N, 'r')
    except:
        poptx = np.empty_like(poptx)
        poptx[:] = np.nan
        pcovx = np.empty_like(pcovx)
        pcovx[:] = np.nan
        popty = np.empty_like(popty)
        popty[:] = np.nan
        pcovy = np.empty_like(pcovy)
        pcovy[:] = np.nan
        
    poptsx.append(poptx)
    pcovsx.append(np.diag(pcovx))
    poptsy.append(popty)
    pcovsy.append(np.diag(pcovy))
    zsb.append(np.mean(df.pos_data[2]))
    zfb.append(np.mean(df.pos_fb[2]))
    Asx.append(poptx[-1])
    Asy.append(popty[-1])
poptsx = np.array(poptsx)
pcovsx = np.array(pcovsx)
poptsy = np.array(poptsy)
pcovsy = np.array(pcovsy)
zfb = np.array(zfb)
zsb = np.array(zsb)*phipbit/4


# In[19]:

psdx


# Out[19]:

#     array([1.12933364e+06, 6.40656933e+05, 4.75974825e+04, ...,
#            2.45166665e+00, 2.35532776e+00, 2.37027361e-01])

# In[11]:

plt.errorbar(zsb, np.abs(poptsx[:, 0])/(2.*np.pi), np.sqrt(pcovsx[:, 0])/(2.*np.pi), fmt = 'o', label = "X 20180927")
plt.errorbar(zsb, np.abs(poptsy[:, 0])/(2.*np.pi), np.sqrt(pcovsy[:, 0])/(2.*np.pi), fmt = 'o', label = "Y 20180927")
#plt.errorbar(zsb2, np.abs(poptsx2[:, 0])/(2.*np.pi), np.sqrt(pcovsx2[:, 0])/(2.*np.pi), fmt = 'o', label = "X 20180906")
#plt.errorbar(zsb07, np.abs(poptsx07[:, 0])/(2.*np.pi), np.sqrt(pcovsx07[:, 0])/(2.*np.pi), fmt = 'o', label = "X 20180907")
plt.legend()
plt.xlabel(r"displacement[$\mu m$]")
plt.ylabel("resonant frequency [Hz]")
plt.ylim([0, 500])
plt.show()


# In[807]:

plt.errorbar(zsb07*4./phipbit, np.abs(poptsx07[:, 0])/(2.*np.pi), np.sqrt(pcovsx07[:, 0])/(2.*np.pi), fmt = 'o', label = "X 20180907")
plt.errorbar(zsb07*4./phipbit, np.abs(poptsy07[:, 0])/(2.*np.pi), np.sqrt(pcovsy07[:, 0])/(2.*np.pi), fmt = 'o', label = "Y 20180907")
plt.show()


# In[775]:

data = np.loadtxt("power_v_bits.csv", skiprows = 2, delimiter = ',')


# In[776]:

power = interpolate.interp1d(data[:, 0], 2.*data[:, 1])


# In[777]:

plt.plot(data[:, 0], 2.*data[:, 1], 'o')
plt.plot(data[:, 0], power(data[:, 0]), 'r')
plt.xlabel("bits")
plt.ylabel("power [mW]")
plt.show()


# In[811]:

b = (zfb<-10000)*(np.abs(zfb)<2**15)
plt.plot(zsb, power(zfb), 'o', label = "20180904")
b2 = (zfb2<-10000)*(np.abs(zfb2)<2**15)
plt.plot(zsb2, power(zfb2), 'o', label = "20180906")
b07 = (zfb07<-10000)*(np.abs(zfb07)<2**15)*(zfb07>-20000)
plt.plot(zsb07[b07], power(zfb07[b07]), 'o', label = "20180907")
plt.legend()
plt.xlabel(r"displacement [$\mu m$]")
plt.ylabel("plev[mW]")
plt.show()


# In[783]:

po = 0.294
plt.plot(zsb2, (po)/power(zfb2), 'o')
plt.xlabel(r"displacement [$\mu m$]")
plt.ylabel("trapping efficiency")
plt.show()


# In[656]:

import re
import glob
cal_files = glob.glob("fzdata_NA/*")
fzs = []
nas = []
zs = []
for f in cal_files:
    data = np.loadtxt(f)
    nums = re.findall(r'\d+', f)
    na = float(nums[0] + '.' + nums[1])
    fzs.append(data[:, 1])
    nas.append(na)
    zs.append(data[:, 0])
inds = np.argsort(np.array(nas))
fzs = np.array(fzs)[inds]
nas = np.array(nas)[inds]
zs = np.array(zs)


# In[657]:

nas = np.array(nas)
fzs = np.array(fzs)
zs = np.array(zs)
b = nas>= 0.085
nas = nas[b]
fzs = fzs[b]
zs = zs[b]
inds  = np.argsort(nas)

for i, fz in enumerate(np.array(fzs)[inds]):
    plt.plot(data[:, 0], fz, label = str(nas[inds[i]]))
plt.legend()
plt.xlabel(r'Axial displacement [$\lambda$]')
plt.ylabel('Axial trapping displacement')
plt.show()


# In[658]:

nas_full = np.einsum('ij,i->ij', np.array(map(np.ones_like, fzs)), nas)


# In[583]:

zs.flatten()


# Out[583]:

#     array([-50.  , -48.99, -47.98, ...,  47.98,  48.99,  50.  ])

# In[608]:

zs[0].shape


# Out[608]:

#     (100,)

# In[670]:

from scipy import interpolate
fz = interpolate.interp2d(zs[0], nas, fzs, kind = 'linear')


# In[671]:

pinds = (np.abs(zs.flatten()-1.52)<.1)
plt.plot(np.linspace(0.09, 0.15, 100), fz(1.52, np.linspace(0.09, 0.15, 100)), 'o')
plt.plot(nas_full.flatten()[pinds], fzs.flatten()[pinds], 'or')
plt.show()


# In[615]:

np.arange(-40, 40, 100)


# Out[615]:

#     array([-40])

# In[710]:

plt.plot(np.linspace(-50, 50, 100), fz(np.linspace(-50, 50, 100), .10), 'o')
plt.plot(np.linspace(-50, 50, 100), fz(np.linspace(-50, 50, 100), .1025), 'o')
plt.plot(np.linspace(-50, 50, 100), fz(np.linspace(-50, 50, 100), .105), 'o')
plt.plot(np.linspace(-50, 50, 100), fz(np.linspace(-50, 50, 100), .1075), 'o')
plt.plot(np.linspace(-50, 50, 100), fz(np.linspace(-50, 50, 100), .11), 'o')
plt.show()


# In[598]:

plt.plot(zs.flatten()[pinds], 'o')
plt.show()


# In[721]:

def fitz(z, na, z0, a):
    return a*fz(z-z0, na)


# In[675]:

z = np.linspace(-30, 30, 60)
plt.plot(z, fitz(z, 0.0975, 0), 'r')
plt.plot(z, fitz(z, 0.1, 0))
plt.plot(z, fitz(z, 0.1025, 0))
plt.show()


# In[676]:

bf = zsb>-55
p0 = [0.11, -25]
popt, pcov = curve_fit(fitz, zsb[bz*bf], (po)/power(zfb[bz*bf]), p0 = p0)


# In[722]:

def cost(params):
    na, z0, a = params
    return np.sum((fitz(zsb[bz*bf], na, z0, a) - (po)/power(zfb[bz*bf]))**2)


# In[742]:

def cost_test(params, ptrue = [0.12, -21., 0.8]):
    na, z0, a = params
    return np.sum((fitz(zsb[bz*bf], *params) - fitz(zsb[bz*bf], *ptrue))**2)


# In[743]:

import scipy.optimize
res  = scipy.optimize.minimize(cost, [0.1, -0, 1], bounds = [[0.09, 0.15], [-30, -10], [0.7, 0.9]])
res2  = scipy.optimize.minimize(cost_test, [0.1, -0, 1], bounds = [[0.09, 0.15], [-30, -10], [0.7, 0.9]])


# In[745]:

res2


# Out[745]:

#           fun: 1.2710667589853745e-12
#      hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
#           jac: array([ 2.31264654e-06, -3.13509966e-09,  1.19767960e-05])
#       message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
#          nfev: 144
#           nit: 24
#        status: 0
#       success: True
#             x: array([  0.12000008, -20.99999664,   0.80000021])

# In[749]:

cz = lambda z: cost([0.111, z, 0.800])
plt.plot(np.linspace(-40, 0, 100), map(cz, np.linspace(-40, 0, 100)))
plt.xlabel(r"displacement [$\lambda$]")
plt.ylabel("cost")
plt.show()


# In[747]:

bz = (zfb<-10000)*(np.abs(zfb)<2**15)
po = 0.294
plt.plot(zsb[bz*bf], (po)/power(zfb[bz*bf]), 'o')
plt.plot(zsb[bz], fitz(zsb[bz], *res.x), label = "NA: " +  str(res.x[0])[:6] + ", z0: "+ str(res.x[1])[:5])
plt.xlabel(r"displacement [$\mu m$]")
plt.ylabel("trapping efficiency")
plt.legend()
plt.show()


# In[678]:

popt
pcov


# Out[678]:

#     array([[2.59713167e-07, 3.25895252e-06],
#            [3.25895252e-06, 1.18133679e-02]])
