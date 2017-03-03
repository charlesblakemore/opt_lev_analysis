import numpy as np
import matplotlib.pyplot as plt

patch_pot = np.loadtxt("patch_pot.txt", delimiter=',', skiprows=1)

patch_pot[:,0] *= 1e-9 ## convert nm to m
patch_pot[:,1] *= 1e-6 ## convert mV^2 to V^2

sep_vec = np.logspace(np.log10(5e-8), -5, 40)
#sep = 2e-6 # gap from shield, in m
epsilon_0 = 8.9e-12 ## SI units
R = 2.5e-6 ## bead radius, m

fvec = np.zeros_like(sep_vec)
for i,sep in enumerate(sep_vec):

    k_vec = np.linspace(np.max([1./patch_pot[-1,0], 1./(10.*sep)]), np.min([10./sep,1./patch_pot[0,0]]),1e7)
    dist_vec = 1./k_vec
    patch_vec = np.interp(dist_vec, patch_pot[:,0], patch_pot[:,1])

    dk = np.median(np.diff(k_vec))

    fs = 4*np.pi*epsilon_0*R/(k_vec[-1]**2 - k_vec[0]**2) * np.sum( patch_vec**2 * k_vec**2 * np.exp(-k_vec*sep)/(np.sinh(k_vec*sep)) * dk)

    print sep, fs

    fvec[i] = fs


fig = plt.figure()
plt.loglog( sep_vec*1e6, fvec, 'k', linewidth=1.5)
xx = plt.xlim()
plt.plot(xx, [5e-20, 5e-20],'k:', linewidth=1.5)
plt.plot(xx, [1.4e-23, 1.4e-23],'k--', linewidth=1.5)
plt.xlabel("Face-to-face separation, $s$ [$\mu$m]")
plt.ylabel("Force due to patch potentials, $F_p$ [N]")
plt.xlim([0.05, 5])
plt.ylim([1e-24, 1e-18])

fig.set_size_inches(5, 3.25)
plt.subplots_adjust(bottom=0.15, top=0.92, left=0.17, right=0.99)
plt.savefig("diff_patch.pdf")

plt.show()

