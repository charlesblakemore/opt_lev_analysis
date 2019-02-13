import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

gap_list = np.array([0.5, 2., 5., 10., 20.])*1e-6
lam_list = np.logspace(-1.0,2.0,40)*1e-6
zoff_list = np.linspace(-100,100,501)*1e-6
zoff_list_new = np.linspace(-1000,1000,5001)*1e-6

fing_width = 10.
rhoau = 19.3e3 # density attractor
rhocu = 8.96e3 # density of copper

contrast_arr = np.zeros( (len(gap_list), len(lam_list) ) )
fmax_arr = np.zeros( (len(gap_list), len(lam_list) ) )

def ffn(x,a,b):
    return a*np.sin(2*np.pi*(x-5.)/10.) + b

for i,gap in enumerate(gap_list):
    for j,lam in enumerate(lam_list):
        
        print gap, lam

        fname = 'data/lam_arr_cu_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)

        cdat = np.load(fname)

        au_force = np.zeros_like(zoff_list_new)
        cu_force = np.zeros_like(zoff_list_new)
        tot_force = np.zeros_like(zoff_list_new)

        for fing_idx in range(-100,101):
            if( fing_idx % 2 == 0):
                au_force += np.interp(zoff_list_new*1e6, zoff_list*1e6 + (fing_idx-0.5)*fing_width, -cdat[:,0])
                tot_force += np.interp(zoff_list_new*1e6, zoff_list*1e6 + (fing_idx-0.5)*fing_width, -cdat[:,0])
            else:
                cu_force += rhocu/rhoau*np.interp(zoff_list_new*1e6, zoff_list*1e6 + (fing_idx-0.5)*fing_width, -cdat[:,0])
                tot_force += rhocu/rhoau*np.interp(zoff_list_new*1e6, zoff_list*1e6 + (fing_idx-0.5)*fing_width, -cdat[:,0])

        min_idx = np.argmin( np.abs(zoff_list_new*1e6 - 5.0) )
        max_idx = np.argmin( np.abs(zoff_list_new*1e6 + 5.0) )
        
        print tot_force[max_idx]-tot_force[min_idx]

        contrast_arr[i,j] = tot_force[max_idx]-tot_force[min_idx]
        fmax_arr[i,j] = np.max(tot_force)

        ## take fft to find component at proper frequency
        xpts = zoff_list*1e6
        gpts = np.logical_and( xpts > -45., xpts < 45. )
        fft = np.abs(np.fft.rfft( tot_force[gpts] ))

        #print abs(fft[5]-fft[3])
        #contrast_arr[i, j] = abs(fft[5]-fft[3])/fft[0]*np.max(tot_force)
        

        # print fft[5]

        # plt.figure()
        # plt.plot(fft)
        # plt.show()


        fig = plt.figure()
        plt.plot(zoff_list_new*1e6,au_force, 'r', linewidth=1.5, label="Au")
        plt.plot(zoff_list_new*1e6,cu_force, 'b', linewidth=1.5, label="Cu")
        plt.plot(zoff_list_new*1e6,tot_force, 'k', linewidth=1.5, label="Tot")

        plt.xlim([-100,100])

        plt.title(r"$\lambda$ = %.3f $\mu$m, gap = %.1f $\mu$m, $\alpha$=1"%(lam*1e6, gap*1e6)) 
        plt.xlabel("Position along array, $x$ [$\mu$m]")
        plt.ylabel("Force towards array, F$_z$, [N]")
        plt.legend()

        fig.set_size_inches(8,6)
        plt.savefig("plots/force_vs_pos_%.3f_%.3f.pdf"%(lam*1e6,gap*1e6))
        
        plt.close()

fig=plt.figure()
for i,gap in enumerate(gap_list):
    plt.loglog(lam_list * 1e6, contrast_arr[i,:], label="gap = %.1f"%(gap*1e6))

plt.legend(loc="lower right")
plt.ylim([1e-30, 1e-20])
plt.xlabel("Yukawa length scale, $\lambda$, [$\mu$m]")
plt.ylabel("Contrast in force towards array, $\Delta$F$_z$, [N]")

fig.set_size_inches(8,6)
plt.savefig("plots/force_contrast_vs_gap.pdf")

fig=plt.figure()
for i,gap in enumerate(gap_list):
    plt.loglog(lam_list * 1e6, 1e-18/contrast_arr[i,:], label="gap = %.1f"%(gap*1e6))

plt.legend(loc="lower left")
plt.ylim([1e0, 1e15])
plt.xlabel("Yukawa length scale, $\lambda$, [$\mu$m]")
plt.ylabel(r"Minimum resolvable $\alpha$, $\sigma_F = 1$ aN")

fig.set_size_inches(8,6)
plt.savefig("plots/alpha_vs_gap.pdf")

fig=plt.figure()
for i,gap in enumerate(gap_list):
    plt.loglog(lam_list * 1e6, contrast_arr[i,:]/fmax_arr[i,:], label="gap = %.1f"%(gap*1e6))

plt.legend(loc="lower left")
#plt.ylim([1e-30, 1e-20])
plt.xlabel("Yukawa length scale, $\lambda$, [$\mu$m]")
plt.ylabel(r"Fractional contrast in force towards array")

fig.set_size_inches(8,6)
plt.savefig("plots/rel_contrast_vs_gap.pdf")

plt.show()
