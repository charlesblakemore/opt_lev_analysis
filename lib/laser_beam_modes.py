import os, sys

import numpy as np
import scipy.special as special
import scipy.ndimage as ndimage
import scipy.integrate as integrate
import unwrap

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 14})





###################################################################################
###################################################################################
###################################################################################



def waist_func(z, w0, zR):
    return w0*np.sqrt( 1 + (z/zR)**2 )

def radius_func(z, zR):
    arr = z * ( 1 + (zR/z)**2 )
    return np.nan_to_num(arr, nan=np.inf)

def guoy_func(z, zR):
    return np.arctan2(z, zR)





def _position_handler(x, y, z):
    '''
    Check if the position arguments are 1D arrays or meshgrid outputs by
    looking at their shape. pretty rudimentary.

    INPUTS:

        x - array-like of floats, with desired x-coordinates. This argument
            can be one-dimensional, or the ouput of a np.meshgrid() call. 
            focus assumed to be at x=y=z=0

        y - array-like of floats, with desired y-coordinates. Same as "x"

        z - array-like of floats, with desired z-coordinates. Same as "x".

    OUTPUTS:

        X - meshgrid for X coordinates

        Y - meshgrid for Y coordinates

        Z = meshgrid for Z coordinates
    '''

    ### Moderate argument handling, mostly for the position vectors/arrays
    x = np.array(x); y = np.array(y); z = np.array(z)
    lens = [len(arr.shape) for arr in [x,y,z]]
    assert all(elem==lens[0] for elem in lens), "x,y,z need same dimensions"
    if lens[0] <= 1:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    else:
        X = np.copy(x)
        Y = np.copy(y)
        Z = np.copy(z)

    return X, Y, Z



def gaussian_complex_field(x, y, z, w0=1e-3, lambda0=1064.0e-9, n_medium=1.0):
    '''
    Compute the electric field of a desired LG mode, sampled over a
    three-dimensional cartesian grid

    INPUTS:

        x - array-like of floats, with desired x-coordinates. This argument
            can be one-dimensional, or the ouput of a np.meshgrid() call. 
            focus assumed to be at x=y=z=0

        y - array-like of floats, with desired y-coordinates. Same as "x"

        z - array-like of floats, with desired z-coordinates. Same as "x".

        n - int, degree of the LG polynomial (number of radial nodes)

        alpha - int, rotational mode number (number of 2pi revolutions)

        w0 - float, gaussian beam waist parameter

        lambda0 - float, wavelength

        n_medium - float, refractive index of the medium

    OUTPUTS:

        field - complex-valued array, electric field of desired LG mode at
            at the requested sampling points

    '''

    X, Y, Z = _position_handler(x, y, z)

    ### A useful set number to have
    zR = np.pi * w0**2 * n_medium / lambda0
    k = 2.0 * np.pi * n_medium / lambda0

    ### Get the polar representation of our sample grid
    r = np.sqrt(X**2 + Y**2)

    ### Compute some paraxial optics quantities
    waist = waist_func(Z, w0, zR)
    radius = radius_func(Z, zR)
    guoy = guoy_func(Z, zR)

    ### Build up the gaussian mode in parts
    geom1 = w0 / waist
    geom2 = np.exp( - r**2 / waist**2 )
    phase1 = np.exp( -1j * k * ( Z + r**2 / (2 * radius) ) )
    phase2 = np.exp( 1j * guoy )

    return geom1 * geom2 * phase1 * phase2




def LG_mode_complex_field(x, y, z, n=0, alpha=0, w0=1e-3, \
                          lambda0=1064.0e-9, n_medium=1.0):
    '''
    Compute the electric field of a desired LG mode, sampled over a
    three-dimensional cartesian grid

    INPUTS:

        x - array-like of floats, with desired x-coordinates. This argument
            can be one-dimensional, or the ouput of a np.meshgrid() call. 
            focus assumed to be at x=y=z=0

        y - array-like of floats, with desired y-coordinates. Same as "x"

        z - array-like of floats, with desired z-coordinates. Same as "x".

        n - int, degree of the LG polynomial (number of radial nodes)

        alpha - int, rotational mode number (number of 2pi revolutions)

        w0 - float, gaussian beam waist parameter

        lambda0 - float, wavelength

        n_medium - float, refractive index of the medium

    OUTPUTS:

        field - complex-valued array, electric field of desired LG mode at
            at the requested sampling points

    '''

    X, Y, Z = _position_handler(x, y, z)

    ### A useful set number to have
    zR = np.pi * w0**2 * n_medium / lambda0

    ### Get the polar representation of our sample grid
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    ### Compute some paraxial optics quantities
    waist = waist_func(Z, w0, zR)
    guoy = guoy_func(Z, zR)

    ### Build up the LG mode in parts
    prefactor = np.sqrt( (2 * np.math.factorial(n)) / \
                         (np.pi * np.math.factorial(n + np.abs(alpha))) )
    geom_lg1 = ( np.sqrt(2)*r / waist )**np.abs(alpha)
    lg = special.genlaguerre(n, alpha)(2*r**2 / waist**2)
    phase_lg1 = np.exp( -1j * alpha * phi)

    ### No +1 in guoy term below because guoy already in the gaussian part
    phase_lg2 = np.exp( 1j * guoy * (np.abs(alpha) + 2*n) )  

    ### Get the gaussian content
    gauss_field = gaussian_complex_field(X, Y, Z, w0=w0, lambda0=lambda0, n_medium=n_medium)

    return prefactor * geom_lg1 * lg * phase_lg1 * phase_lg2 * gauss_field




def diffraction_circular_aperture(complex_field_func, aperture_radius, z=0, \
                                  view_dist=1.0, theta_max=0, npts=1001, \
                                  lambda0=1064.0e-9, field_args={}, **kwargs):


    ### DOESN'T WORK IF THERE IS ANY PHI DEPENDENCE IN THE INPUT FIELD /
    ### APERTURE FUNCTION. INTEGRAL IS MORE COMPLEX IF THAT'S THE CASE.
    ### THIS IS LEFT AS A FURTHER EXERCISE IF DEEMED NECESSARY

    field_args['lambda0'] = lambda0

    ### 10x the first minima from the Fraunhoffer condition
    if theta_max == 0:
        theta_max = 10 * lambda0 / aperture_radius

    theta_arr = np.linspace(0, theta_max, npts)

    radial_profile = np.zeros(npts, dtype=np.float)
    # radial_profile = np.zeros(npts, dtype=np.complex128)

    for theta_index, theta in enumerate(theta_arr):

        def integrand(rho_prime):
            aperture = np.abs(complex_field_func([rho_prime], [0], [0], **field_args))
            bessel = special.jv(0, 2.0*np.pi*np.sin(theta)*rho_prime/lambda0)
            return aperture * bessel * rho_prime

        integral = integrate.quad(integrand, 0, aperture_radius)

        radial_profile[theta_index] = integral[0]

    theta_twoside = np.concatenate((-1.0*theta_arr[::-1], theta_arr[1:]))
    profile_twoside = np.concatenate((radial_profile[::-1], radial_profile[1:]))

    return theta_twoside, profile_twoside






def second_moment(x, y, field_slice, axis=0, gblur=False, sigma_blur=1):
    lens = [len(arr.shape) for arr in [x,y]]
    assert all(elem==lens[0] for elem in lens), "x,y,z need same dimensions"
    if lens[0] <= 1:
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        X = np.copy(x)
        Y = np.copy(y)

    intensity = np.abs(field_slice)**2
    if gblur:
        dx = X[1,0] - X[0,0]
        dy = Y[0,1] - Y[0,0]
        blur_kernel = sigma_blur / np.mean([dx, dy])
        intensity = ndimage.gaussian_filter(intensity, sigma=blur_kernel)

    if axis == 0:
        return np.sqrt( np.sum(X**2 * intensity) / np.sum(intensity)  )
    else:

        return np.sqrt( np.sum(Y**2 * intensity) / np.sum(intensity)  )






def plot_field_slice(x, y, field_slice, gblur=False, sigma_blur=1, \
                     cmap='plasma', show=True, **kwargs):

    fig, axarr = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))

    intensity = np.abs(field_slice)**2
    phase = unwrap.unwrap(np.angle(field_slice))
    if gblur:
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        blur_kernel = sigma_blur / np.mean([dx, dy])
        intensity = ndimage.gaussian_filter(intensity, sigma=blur_kernel)
        phase = ndimage.gaussian_filter(phase, sigma=blur_kernel)

    min_phase = np.floor(np.min(phase)/np.pi)
    max_phase = np.ceil(np.max(phase)/np.pi)
    if min_phase == max_phase:
        max_phase += 1.0
    phase_levels = np.linspace(min_phase, max_phase, 501)*np.pi

    kwargs['cmap'] = cmap
    intensity_contours = axarr[0].contourf(y*1e6, x*1e6, intensity, \
                                           501, **kwargs)
    phase_contours = axarr[1].contourf(y*1e6, x*1e6, phase, \
                                       levels=phase_levels, **kwargs)

    phase_ticks = []
    phase_ticklabels = []
    for i in range(int(max_phase - min_phase)+1):
        phase_val = min_phase + i
        phase_ticks.append(phase_val*np.pi)
        if not phase_val:
            phase_ticklabels.append('0')
        elif phase_val == 1:
            phase_ticklabels.append('$\\pi$')
        elif phase_val == -1:
            phase_ticklabels.append('$-\\pi$')
        else:
            phase_ticklabels.append(f'{int(phase_val):d}$\\pi$')

    axarr[0].set_xlim(np.min(x)*1e6, np.max(x)*1e6)
    axarr[0].set_ylim(np.min(y)*1e6, np.max(y)*1e6)
    for i in [0,1]:
        axarr[i].set_aspect('equal')
        axarr[i].set_xlabel('X-coord [um]')
        
    axarr[0].set_ylabel('Y-coord [um]')

    axarr[0].set_title('Intensity')
    axarr[1].set_title('Phase')

    fig.tight_layout()

    fig.subplots_adjust(right=0.925)

    ### Same thing for the phase
    phase_inset = inset_axes(axarr[1], width="4%", height="85%", \
                             loc='center right', \
                             bbox_to_anchor=(0.07, 0, 1, 1), \
                             bbox_transform=axarr[1].transAxes, \
                             borderpad=0)

    phase_cbar = fig.colorbar(phase_contours, cax=phase_inset, ticks=phase_ticks)
    phase_cbar.ax.set_yticklabels(phase_ticklabels)

    if show:
        plt.show()

    return fig, axarr




def plot_marginalized_profile(x, y, field_slice, axis=0, \
                              gblur=False, sigma_blur=1, show=True, \
                              fig=None, ax=None, **kwargs):

    if fig is None:
        fig, ax = plt.subplots(1,1)

    intensity = np.abs(field_slice)**2
    measures = [ x[1]-x[0], y[1]-y[0] ]
    if gblur:
        blur_kernel = sigma_blur / np.mean(measures)
        intensity = ndimage.gaussian_filter(intensity, sigma=blur_kernel)

    if axis == 0:
        vec = np.copy(x)
        profile = np.sum(intensity, axis=1) * measures[1]
    else:
        vec = np.copy(y)
        profile = np.sum(intensity, axis=0) * measures[0]

    profile /= np.max(profile)

    ax.plot(vec*1e6, profile, **kwargs)
    ax.set_yscale('log')
    ax.set_ylim(3e-4, 1.2)
    ax.set_ylabel('Marginalized Irradiance [arb.]')
    if axis == 0:
        ax.set_xlabel('X-coord [um]')
    else:
        ax.set_xlabel('Y-coord [um]')
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

