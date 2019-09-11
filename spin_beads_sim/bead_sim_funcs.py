######################################################
# Functions for use in bead spinning simulations. 
# Has a few constants as well.
######################################################


import sys
import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torsion_noise as tn

import time

### Constants
e = 1.602e-19          #  C
p0 = 100 * e * (1e-6)  #  C * m
rhobead = 1550
rbead = 2.4e-6
mbead = (4. / 3.) * np.pi * rbead**3 * rhobead
Ibead = (2. / 5.) * mbead * rbead**2 


### Set the thermalization time 
therm_time = 50    # s


def chirpE(ti, tf, dt, fieldmag, chirp_start, chirp_end, chirp_length, \
            xchirp=False, ychirp=False, zchirp=True, xphi=-90, yphi=0, zphi=0, \
            steady=True, twait=0.0, tmin=0.0, tmax=1e9):
    '''Generates an E-field chirp along any of three directions, with
       and independent phase for each direction in order to induce rotation.'''

    # Array at which to evaluate field (simulation points)
    tarr = np.arange(ti, tf+dt, dt)
    tbool = (tarr >= tmin) * (tarr <= tmax)

    # Make some boolean arrays to turn fields on and off, since scipy.signal's
    # definition of chirp is over the interval t in (-inf, +inf)
    chirp_bool = (tarr >= (therm_time + twait)) * \
                    (tarr < (therm_time + twait + chirp_length))
    steady_bool = tarr >= (therm_time + twait + chirp_length)

    # Look-up dictionary for each axis
    dic = {0: (xchirp, xphi), 1: (ychirp, yphi), 2: (zchirp, zphi)}

    Eout = [[], [], []]
    for ind in [0,1,2]:
        # Find whether to chirp and with what phase
        chirp, phi = dic[ind]

        # Generate the chirp!
        Echirp = fieldmag * signal.chirp(tarr-therm_time-twait, 
                                    chirp_start, chirp_length, chirp_end, phi=phi)

        # Generate a constant frequency steady-state field after 
        # the chirp (or zeros if no steady-state desired)
        if steady:
            Esteady = fieldmag * np.cos(2 * np.pi * chirp_end * \
                                    (tarr - (therm_time + chirp_length + twait)) \
                                    + np.pi * phi / 180.0)
        elif not steady:
            Esteady = np.zeros_like(tarr)

        # Build the full chirp e-field, with a steady-state
        Echirp_full = Echirp * chirp_bool + Esteady * steady_bool

        # Set the field to 0 outside of [tmin, tmax]
        if chirp:
            Eout[ind] = Echirp_full * tbool
        elif not chirp:
            Eout[ind] = np.zeros_like(tarr)

    return np.array(Eout)

def oscE(ti, tf, dt, fieldmag, xfreq, yfreq, zfreq, \
            xphi=0.0, yphi=0.0, zphi=0.0, phase_mod=False,\
            x_mod_amp=0.0, y_mod_amp=0.0, z_mod_amp=0.0,\
            x_mod_freq=0.0, y_mod_freq=0.0,z_mod_freq=0.0,\
            tmin=0.0, tmax=1e9):
    '''Generates a oscillating E-field along any of three directions,
    with capability of phase modulating efield.'''

    # Array at which to evaluate field (simulation points)
    tarr = np.arange(ti, tf+dt, dt)
    tbool = (tarr >= tmin) * (tarr <= tmax)

    # Look-up dictionary for each axis
    dic = {0: (xfreq, xphi), 1: (yfreq, yphi), 2: (zfreq, zphi)}
   
    Eout = [[], [], []]
    if phase_mod:
       
        mod_dic = {0: x_mod_freq, 1: y_mod_freq, 2: z_mod_freq}
        
        for ind in [0,1,2]:
            # Find whether to oscillate and with what phase
            freq, phi = dic[ind]
            mod_freq = mod_dic[ind]

            # Build field
            if freq:
                mod = 0.
                
                if mod_freq:
                    mod = x_mod_amp * np.cos(2 * np.pi * mod_freq * tarr)
                
                Eosc = fieldmag * np.cos(2 * np.pi * freq * tarr + mod + 
                                                        np.pi * phi / 180.0)
                Eout[ind] = Eosc
                # Set to 0 outside of [tmin, tmax]
                Eout[ind] = Eosc * tbool
            elif not freq:
                Eout[ind] = np.zeros_like(tarr)
    

    else:
        for ind in [0,1,2]:
            # Find whether to oscillate and with what phase
            freq, phi = dic[ind]

            # Build field
            if freq:
                Eosc = fieldmag * np.cos(2 * np.pi * freq * tarr + 
                                                    np.pi * phi / 180.0)
                Eout[ind] = Eosc
                # Set to 0 outside of [tmin, tmax]
                Eout[ind] = Eosc * tbool
            elif not freq:
                Eout[ind] = np.zeros_like(tarr)

    return np.array(Eout)

def step_func(ti, tf, dt, mag, step_offset=0.):
    tarr = np.arange(ti, tf+dt, dt)

    func = mag*np.ones_like(tarr)
    
    func[tarr < step_offset] = 0.

    return func

def constant_field(ti, tf, dt, fieldmag, x=False, y=False, \
        z=False):
    tarr = np.arange(ti, tf+dt, dt)

    Eout = [[],[],[]]

    dic = {0: x, 1: y, 2: z}

    for ind in [0,1,2]:
        if dic[ind]:
            Eout[ind] = fieldmag * np.ones_like(tarr)
        
        else:
            Eout[ind] = np.zeros_like(tarr)

    return np.array(Eout)




def rk4(xi_old, t, tind, delt, system):
    '''4th-order Runge-Kutta integrator method. Takes an input vector
       of the form [x_i, y_i, ..., dx_i/dt, dy_i/dt, ...] and uses a 
       user-defined system which computes the derivatives of each of 
       the coordinates and their corresponding velocities at the given
       instant in time. Essentially, the function 'system' returns:
       [dx_i/dt, dy_i/dt, ..., d^2x_i/dt^2, d^2y_i/dt^2, ...], where 
       the acceleration terms are determined by external fields and 
       thermal fluctuations and such.'''
    k1 = delt * system(xi_old, t, tind)
    k2 = delt * system(xi_old + k1 / 2, t + (delt / 2), tind)
    k3 = delt * system(xi_old + k2 / 2, t + (delt / 2), tind)
    k4 = delt * system(xi_old + k3, t + delt, tind)
    xi_new = xi_old + (1. / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    #raw_input()
    ### HARDCODED STUFF IS BAD BUT HERE IT IS ANYWAY
    # Correction to keep dipole magnitude normalized, or small integration 
    # build up. This is only useful for the spcific implementation
    # of electrostatically spinning beads
    ptot = np.sqrt(xi_new[6]**2 + xi_new[7]**2 + xi_new[8]**2)
    wtot = np.sqrt(xi_new[0]**2 + xi_new[1]**2 + xi_new[2]**2)
    
    #print ptot, wtot
    for ind in [6,7,8]:
        xi_new[ind] *= p0 / ptot
    

    return xi_new



def stepper(xi_0, ti, tf, delt, system, method, efield):
    '''This function does the full integration. Takes a system function 
    (which is usually defined in the simulation script since it's subject
    to change), an integration method, an external efield to keep track 
    of the dipole's energy, and of course initial conditions.'''

    # Build the array of solution times
    tt = np.arange(ti, tf + delt, delt)

    # Initialize list of 'points' which will contain the solution to our
    # ODE at each time in our discrete-time array
    points = []
    energy_vec = []
    i = 0
    xi_old = xi_0

    ticker = 0

    for tind, t in enumerate(tt):
        start_t = time.time()
        # Using the 4th-order Runge-Kutta method, we evaluate the solution
        # iteratively for each time t in our discrete-time array
        xi_new = method(xi_old, t, tind, delt, system)

        points.append(xi_new)

        Ex, Ey, Ez = efield[:,tind]
        # Compute energy as:  (1/2) I omega^2 - p (dot) E
        #energy = 0.5 * Ibead * (xi_new[3]**2 + xi_new[4]**2 + xi_new[5]**2) - \
        #            (Ex * xi_new[0] + Ey * xi_new[1] + Ez * xi_new[2])

        #energy_vec.append(energy)

        xi_old = xi_new

        if (t / tf) > (i * 0.01):
            print i,
            sys.stdout.flush()
            i += 1
        ticker += 1
        stop_t = time.time()
        #print "Loop time: ", stop_t - start_t

    return tt, np.array(points), np.array(energy_vec)


