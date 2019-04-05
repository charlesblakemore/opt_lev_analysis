import math
import numpy as np
import matplotlib.pyplot as plt



gap_list = [5.0e-6, 7.5e-6, 10e-6]
lam_list = np.logspace(-1.0,2.0,20)*1e-6
print lam_list
sens_vals_num = np.zeros((len(lam_list),len(gap_list)))

for i in range(len(gap_list)):
    for j in range(4,len(lam_list)):

        gap = gap_list[i]
        lam = lam_list[j]

        fname = 'data/lam_arr_cyl_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
        cval = np.load(fname)

        sigf = 1.9e-16

        sens_vals_num[j,i] = sigf/cval[0]


f0 = 1e3
m = 1e-13

xvals = np.linspace(-2e-6,2e-6,1e3)

harm_pot = 0.5*m*(2*math.pi*f0)**2 * xvals**2

## now assume point mass at distance d from origin
d = 10e-6
Ma = 10e-13
alpha = 1.0e16
lam = 10e-6
G = 6.67e-11

grav_pot = alpha*G*m*Ma * (2.0*np.exp(-(d/lam))/d - np.exp(-np.abs(d-xvals)/lam)/np.abs(d-xvals) - np.exp(-np.abs(-d-xvals)/lam)/np.abs(-d-xvals)) 

grav_pot_approx = -2*alpha*G*m*Ma/d**3*np.exp(-d/lam)*(1 + d/lam + 0.5*(d/lam)**2)*xvals**2

print (1 + d/lam + 0.5*(d/lam)**2)

## now fit to a quadratic term near the minimum
fit_win = [400,600]
p1 = np.polyfit(xvals[fit_win[0]:fit_win[1]], harm_pot[fit_win[0]:fit_win[1]],2)
print p1

tot_pot = harm_pot + grav_pot
p2 = np.polyfit(xvals[fit_win[0]:fit_win[1]], tot_pot[fit_win[0]:fit_win[1]],2)
print p2

fig = plt.figure(33)
plt.plot(xvals,harm_pot)
plt.plot(xvals,harm_pot + grav_pot,'r')

xx = xvals[fit_win[0]:fit_win[1]]
plt.plot(xx,np.polyval(p1,xx),'c')
plt.plot(xx,np.polyval(p2,xx),'m')

fig2 = plt.figure(34)
plt.plot(xvals,grav_pot)
plt.plot(xvals,grav_pot_approx,'r')

plt.show()
