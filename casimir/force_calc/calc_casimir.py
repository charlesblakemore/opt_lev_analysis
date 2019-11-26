## calculate casimir force between two infinite plates, and find corresponding
## supression factor
import glob, re
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

omega_p = 1.4e16 ## Au, rad/s
gamma = 5.3e13 ## Au, rad/s

omega_p_Cu = omega_p ## same as gold
gamma_Cu = 4.6e13 ## slightly smaller

D1 = 5e-6 ## mirror thickness, m
D1_is_Au = False
D_s = 1.5e-6 ## shield thickness, m
D_a = 200.0e-6 ## attractor thickness, m
L = 0.5e-6 ## cavity length, i.e. face-to-face separation, m
c = 3e8 ## m/s
hbar = 1.05e-34 ## J s
G = 6.67e-11

tot_thick_list = np.logspace(-7, np.log10(10e-6), 80)
L_list = [0.2e-6, 0.5e-6, 1e-6, 2e-6]
#tot_thick_list = L_list

def int_func(Omeg, K):
    
    omega = Omeg*c/L
    kappa = K/L
    
    eps_iw = (1 + omega_p**2/(omega*(omega + gamma)))
    
    rho_perp = -(np.sqrt( omega**2 * (eps_iw - 1) + c**2 * kappa**2) - c*kappa)/(np.sqrt( omega**2 * (eps_iw - 1) + c**2 * kappa**2) + c*kappa)
    rho_par = -(np.sqrt( omega**2 * (eps_iw - 1) + c**2 * kappa**2) - c*kappa*eps_iw)/(np.sqrt( omega**2 * (eps_iw - 1) + c**2 * kappa**2) + c*kappa*eps_iw)

    del0 = np.sqrt( omega**2 * (eps_iw - 1) + c**2 * kappa**2 )
    if( D1_is_Au ):
        delta1 = D1/c * del0
    else:
        eps_si = 2.0
        rho_perp_si = -(np.sqrt( omega**2 * (eps_si - 1) + c**2 * kappa**2) - c*kappa)/(np.sqrt( omega**2 * (eps_si - 1) + c**2 * kappa**2) + c*kappa)
        rho_par_si = -(np.sqrt( omega**2 * (eps_si - 1) + c**2 * kappa**2) - c*kappa*eps_si)/(np.sqrt( omega**2 * (eps_si - 1) + c**2 * kappa**2) + c*kappa*eps_si)
        delta1 = D1/c * np.sqrt( omega**2 * (eps_si - 1) + c**2 * kappa**2 )
    delta2 = D2/c * del0

    r1_perp = rho_perp*(1 - np.exp(-2*delta1))/(1-rho_perp**2*np.exp(-2*delta1))
    r1_par = rho_par*(1 - np.exp(-2*delta1))/(1-rho_par**2*np.exp(-2*delta1))
    r2_perp = rho_perp*(1 - np.exp(-2*delta2))/(1-rho_perp**2*np.exp(-2*delta2))
    r2_par = rho_par*(1 - np.exp(-2*delta2))/(1-rho_par**2*np.exp(-2*delta2))

    return 120/np.pi**4 * K**2 * ( r1_perp*r2_perp/(np.exp(2*K)-r1_perp*r2_perp)
                    + r1_par*r2_par/(np.exp(2*K)-r1_par*r2_par) )

## casimir force for 5 um thick at 5 um
D_s = 0e-6
D_a = 10e-6
L = 1e-6
D2 = D_s
cint, err = integrate.dblquad(int_func, 0., 20, lambda x: 0, lambda x: x, epsabs=1e-8, epsrel=1e-8)
Fcas = cint * np.pi**3 * D1/2.0 * hbar * c/(360*L**3)
print("Casimir force: ", Fcas)


if(True):
    out_mat = np.zeros((len(tot_thick_list), len(L_list)))
    for j, tot_thick in enumerate(tot_thick_list):
        print("Working on tot_thick: ", tot_thick)
        for i,L in enumerate(L_list):
            D_s = tot_thick - L
            if( D_s <= 0 ): continue

            D2 = D_s
            cint, err = integrate.dblquad(int_func, 0., 20, lambda x: 0, lambda x: x, epsabs=1e-8, epsrel=1e-8)
            D2 = D_s+D_a
            cint2, err2 = integrate.dblquad(int_func, 0., 20, lambda x: 0, lambda x: x, epsabs=1e-8, epsrel=1e-8)

            Fcas = cint2 * np.pi**3 * D1/2.0 * hbar * c/(360*L**3)
            Fdiffcas = (cint2-cint) * np.pi**3 * D1/2.0 * hbar * c/(360*L**3)

            print("For L=%.1f um, D_s = %.1f um: F_cas = %.3e, dF_cas = %.3e" % (L*1e6, D_s*1e6, Fcas, Fdiffcas))

            out_mat[j,i] = Fdiffcas

    #print "For D1=%.1f um, D2=%.1f um, L=%.1f um: eta=%.10f +/- %.10f"%(D1*1e6, D2*1e6, L*1e6, cint, err)
    np.save("cas_mat.npy", out_mat)
else:
    out_mat = np.load("cas_mat.npy")

out_mat = np.array(out_mat)

fig = plt.figure()
for i in range( len(L_list) ):
    cdat = out_mat[:,i]
    gpts = cdat > 0

    if(i == 1):
        fit_pts = np.logical_and(tot_thick_list > 1e-6, tot_thick_list < 6e-6)
        p = np.polyfit( np.log10(tot_thick_list[fit_pts]), np.log10(cdat[fit_pts]), 1)
        cdat[tot_thick_list > 3.7e-6] = 10**np.polyval(p, np.log10(tot_thick_list[tot_thick_list > 3.7e-6]))
    if(i == 2):
        fit_pts = np.logical_and(tot_thick_list > 3e-6, tot_thick_list < 7e-6)
        p = np.polyfit( np.log10(tot_thick_list[fit_pts]), np.log10(cdat[fit_pts]), 1)
        cdat[tot_thick_list > 5e-6] = 10**np.polyval(p, np.log10(tot_thick_list[tot_thick_list > 5e-6]))

    plt.loglog(tot_thick_list[gpts]*1e6, cdat[gpts], linewidth=1.5, label="$s = %.1f\ \mu\mathrm{m}$"%(L_list[i]*1e6))

xx = plt.xlim()
plt.plot(xx, [5e-20, 5e-20],'k:', linewidth=1.5)
plt.plot(xx, [1.4e-23, 1.4e-23],'k--', linewidth=1.5)

## overplot 1/r^2
m1 = 4./3*np.pi*(2.5e-6)**3*2e3
m2 = (10e-6)**3 * 19.3e3
r = tot_thick_list + (2.5e-6+5e-6)  ##center of the cube
F = G*m1*m2/r**2
plt.plot(tot_thick_list*1e6, F, 'k', linewidth=1.5)

plt.ylim([1e-25, 1e-12])



# ## now for a given lambda, plot the force versus separation
# lam_list = [0.5, 1, 2, 5, 10]
# for lam in lam_list:
#     flist = glob.glob('data/lam_arr_depth_10.000_shield_0.000_gap_*_lam_%.3f.npy'%lam)
#     xvals = []
#     yvals = []
#     for f in flist:
#         cval = np.load(f)
#         gap_val = float(re.findall("gap_\d+",f)[0][4:])
#         xvals.append( gap_val )
#         yvals.append( cval[0] )
#     if(len(xvals)>0):
#         xvals, yvals = zip(*sorted(zip(xvals,yvals)))
#     plt.plot( xvals, yvals, '--')

plt.xlabel("Total separation from attractor, $s+t$ [$\mu$m]")
plt.ylabel("Differential Casimir force [N]")
plt.legend(prop={"size": 13})

fig.set_size_inches(5, 3.75)
plt.subplots_adjust(bottom=0.13, top=0.95, left=0.15, right=0.97)
plt.savefig("diff_casimir.pdf")

plt.show()

