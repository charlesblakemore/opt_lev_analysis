######################################################
# Bead spinning simulation script. Does single-threaded
# integration of a dipole subject to thermal torques
# and an external electric field
######################################################


import sys
import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torsion_noise as tn
import bead_sim_funcs as bsfuncs

### Define the savepath
#savepath = '/spinsim_data/alldata_Vxy100Vrotchirp_1kHz_dt5us.npy'
savepath = '/home/dmartin/Desktop/simulations/added_thermtorque/'
savename = 'test4'

# Random seed. Un-comment to use a fixed seed for numpy's
# quasi-random random number generator in order to check 
# deterministic behavior etc. 
#np.random.seed(123456)

#damping time
tau = 500.


### Get the dipole moment and moment of inertia
p0 = bsfuncs.p0
Ibead = bsfuncs.Ibead
print p0, Ibead

### Define Initial Conditions
theta0 = 0.5 * np.pi # rad
phi0 = 0.0 #0.5            # rad
p0x = p0 * np.sin(theta0) * np.cos(phi0)
p0y = p0 * np.sin(theta0) * np.sin(phi0)
p0z = p0 * np.cos(theta0) * 0.0

wx = 0.0        
wy = 0.0  
wz = 0.0

d = p0 
w = 50e2 * np.pi

xi_init = np.array([p0x, p0y, p0z, wx, wy, wz])

wx0 = 0.#2 * np.pi * 2.e3
wy0 = 0.
wz0 = 0#2 * np.pi * 50.

wxdot0 = 0.
wydot0 = 0.
wzdot0 = 0.


xi_init = np.array([wx0,wy0,wz0,wxdot0,wydot0,wzdot0,p0x, \
        p0y,p0z])

### Integration parameters

# dt = 1.0e-5
dt = 1.0e-3      # time-step, s
ti = 0          # initial time, s
tf = 100    # final time, s

### Drive parameters
maxvoltage = 10.0
fieldmag = maxvoltage / 4.0e-3
drvmag = 10.0 * maxvoltage / 4.0e-3


tt = np.arange(ti, tf + dt, dt)
Nsamp = len( tt )
Nsamp = 12000000
#
## Compute sampling frequency to limit bandwidth of white noise
Fsamp = 1.0 / dt
Fsamp = 50000.

tt = np.arange(0, Nsamp/Fsamp, 1./Fsamp)
ti = tt[0]
tf = tt[-1]
dt = tt[1]-tt[0]

print ti, tf, dt
#### NORMALIZATION OF TORQUE NOISE COULD BE OFF, SHOULD
#### REALLY LOOK INTO THIS MORE
# Compute torque noise from Alex's functional library
thermtorque = np.array([tn.torqueNoise(Nsamp, Fsamp), \
                        tn.torqueNoise(Nsamp, Fsamp), \
                       tn.torqueNoise(Nsamp, Fsamp)])


# Compute damping coefficient from Alex's functional library
beta = tn.beta()



#xchirp = bsfuncs.chirpE(ti, tf, dt, drvmag, 1, 1000, 100, \
#                        xchirp=True, ychirp=True, zchirp=False, \
#                        steady=True, xphi=-90, yphi=0, twait=0, tmax=250)


#efield = xchirp

drvmag = 64.e3#16.5e1
xfreq = 1000.
yfreq = 1000.
zfreq = 0.

xphi = 0.
yphi = -90.
zphi = 0.

x_mod = 2. * np.pi
x_mod_freq = 34.
y_mod = 2. * np.pi
y_mod_freq = 34.

efield = bsfuncs.oscE(ti, tf, dt, drvmag,\
        xfreq=xfreq, yfreq=yfreq, zfreq=zfreq, yphi=yphi,\
        phase_mod=False, x_mod_amp=x_mod, y_mod_amp=x_mod,\
        x_mod_freq=x_mod_freq,y_mod_freq=y_mod_freq)

tarr = np.arange(ti,tf+dt,dt)
mask = tarr > 50

step_arr = bsfuncs.step_func(ti,tf,dt,1.,20)
#step_arr[mask] -= drvmag

efield = np.array([step_arr*efield[0],step_arr*efield[1],np.zeros_like(tarr)])
#efield = np.array([np.zeros_like(tarr), np.zeros_like(tarr), np.zeros_like(tarr)])

#Order of maginitude optical torque
t_opt = 1.e-24

par_dic = {'efieldx':(xfreq,xphi),\
           'efieldy':(yfreq,yphi),\
           'efieldz':(zfreq,zphi),\
           'efield_mag':drvmag,\
           'theta0' :theta0,\
           'phi0'   :phi0,\
           'wx0'    :wx0,\
           'wy0'    :wy0,\
           'wz0'    :wz0,\
           'wxdot0' :wxdot0,\
           'wydot0' :wydot0,\
           'wzdot0' :wzdot0,\
           'p0'     :p0,\
           'tau'    :tau,\
           'Ibead'  :Ibead,\
           'tau_opt':t_opt,\
           'Nsamp'  :Nsamp,\
           'Fsamp'  :Fsamp} 

### UNCOMMENT THESE LINES TO LOOK AT EFIELD BEFORE SIMULATING

fig, axarr = plt.subplots(3,1,sharex=True,sharey=True)

start_time = bsfuncs.therm_time+100 - 10000*dt
x = int(start_time / dt)

for i in [0,1,2]:
    axarr[i].plot(tt, efield[i])
   #axarr[i].plot(tt[x:x+100000], efield[i,x:x+100000])
plt.show()

#print(Nsamp)
#
#fft = np.fft.rfft(efield)
#freqs = np.fft.rfftfreq(Nsamp, 1./Fsamp)
#fig, axarr = plt.subplots(3,1,sharex=True,sharey=True)
#for i in [0,1,2]:
#    axarr[i].loglog(freqs,np.abs(fft[i]))
#
#plt.show()
    
raw_input()




# Keeping this function in the integrator script as it depends on the 
# user-defined electric field and torque noise etc.

def system(xi, t, tind):
    '''Function returns the first order derivatives of the vector
    xi, assuming xi = (px, py, pz, omegax, omegay, omegaz), i.e. the 
    orientation of the dipole and angular momentum of the microsphere.'''

    Ex = efield[0,tind]
    Ey = efield[1,tind]
    Ez = efield[2,tind]
    

    tx = thermtorque[0,tind]
    ty = thermtorque[1,tind]
    tz = thermtorque[2,tind]

    px = xi[0]
    py = xi[1]
    pz = xi[2]

    wx = xi[3]
    wy = xi[4]
    wz = xi[5]
	
    # Consider p vector to be like radial vector and thus given angular
    # velocities defined about the origin, the components change as:
    dpx = py * wz - pz * wy
    dpy = pz * wx - px * wz
    dpz = px * wy - py * wx

    # Compute torque as p (cross) E + thermal torque - damping
    torque = [py * Ez - pz * Ey + tx - wx * beta,  \
              pz * Ex - px * Ez + ty - wy * beta,  \
              px * Ey - py * Ex + tz - wz * beta]
    
    return np.array([-dpx, -dpy, -dpz, torque[0] / (Ibead), \
                        torque[1] / (Ibead), torque[2] / (Ibead)])

def testSystem(xi, t, tind):
    ''' Damped harmonic oscillator'''
    x = xi[0]
    x_dot = xi[1]
    
    dx0 =  x_dot
    dx1 = -x #- x_dot

    return np.array([dx0, dx1])    

def dipole_efield(xi, t, tind):
    theta = xi[0]
    theta_dot = xi[1]
    
    dtheta0 = theta_dot
    dtheta1 = (d/Ibead)  * (30e3) * (np.sin(w * t) * np.cos(theta) - np.cos(w * t)*np.sin(theta))  - (1/2000) * theta_dot

    return np.array([dtheta0, dtheta1])

def dipole_efield_cart(xi, t, tind):
    px = xi[6]
    py = xi[7]
    pz = xi[8]

    tx = thermtorque[0,tind]
    ty = thermtorque[1,tind]
    tz = thermtorque[2,tind]

    Ex = efield[0,tind]
    Ey = efield[1,tind]
    Ez = efield[2,tind]

    omegax = xi[0]
    omegay = xi[1]
    omegaz = xi[2]
    
    omegax_dot = xi[3] 
    omegay_dot = xi[4]
    omegaz_dot = xi[5]

    dpx = py * omegaz - pz * omegay
    dpy = pz * omegax - px * omegaz
    dpz = px * omegay - py * omegax

    domegax_0 =  tx/Ibead + (1/Ibead) * (py*Ez - pz*Ey) - (beta/Ibead) * omegax
    domegay_0 =  ty/Ibead + (1/Ibead) * (pz*Ex - px*Ez) - (beta/Ibead) * omegay
    domegaz_0 = tz/Ibead + (1/Ibead) * (px*Ey - py*Ex) - (beta/Ibead) * omegaz \
            + t_opt/Ibead

    domegax_1 = 0.
    domegay_1 = 0.
    domegaz_1 = 0.
    #Negative on dipole components following Goldstein convention       of ccw rotation (check this)
    return np.array([domegax_0, domegay_0, domegaz_0,\
            domegax_1, domegay_1, domegaz_1, -dpx, -dpy, -dpz]) 
   
    

time, points, energy_vec = bsfuncs.stepper(xi_init, ti, tf, dt,\
        dipole_efield_cart, bsfuncs.rk4, efield=efield)

print points.shape



#outpoints = np.c_[time, points, energy_vec, efield[0,:], efield[1,:], efield[2,:]]

outpoints = np.c_[time, points, efield[0,:], efield[1,:], efield[2,:]]

np.save(savepath + savename + '_data', outpoints)
np.save(savepath + savename + '_parameters', par_dic)




