import os, sys

import numpy as np
import scipy.special as special
import unwrap

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import laser_beam_modes as beams

plt.rcParams.update({'font.size': 14})


z = 0                      # axial position along beam

n_medium = 1.0             # refractive index
lambda0 = 1064.0e-9        # wavelength
n = 1                      # degree of LG mode
alpha = 0                  # 'parameter' of LG mode
w0 = 6.3e-6                # spot_size

### NOTE: sometimes (n,alpha) is (p,l). alpha is an integer for the paraxial
### optical mode solutions

xmin = -40.0e-6
xmax = 40.0e-6
ymin = -40.0e-6
ymax = 40.0e-6
npts = 201

show_naive_waist = False

sigma_blur = 1.0e-9

###################################################################################
###################################################################################
###################################################################################


testX = np.linspace(xmin, xmax, npts)
testY = np.linspace(ymin, ymax, npts)
testZ = [z]

X, Y, Z = np.meshgrid(testX, testY, testZ, indexing='ij')
field0 = beams.LG_mode_complex_field(X, Y, Z, n=0, alpha=0, w0=w0, \
                                     lambda0=lambda0, n_medium=n_medium)
field1 = beams.LG_mode_complex_field(X, Y, Z, n=1, alpha=0, w0=w0, \
                                    lambda0=lambda0, n_medium=n_medium)
field2 = beams.LG_mode_complex_field(X, Y, Z, n=2, alpha=0, w0=w0, \
                                     lambda0=lambda0, n_medium=n_medium)

focal_plane_z_index = np.argmin(np.abs(testZ))
i = focal_plane_z_index

coherent_field_slice = field0[:,:,i] + field1[:,:,i] + field2[:,:,i]
incoherent_field_slice = np.abs(field0[:,:,i]) + np.abs(field1[:,:,i]) #+ 0.25*np.abs(field2[:,:,i])
incoherent_field_slice2 = np.abs(field0[:,:,i]) + 0.5*np.abs(field1[:,:,i]) #+ 0.25*np.abs(field2[:,:,i])

# print(np.abs(field1[:,:,i]))

fig, axarr = beams.plot_field_slice(testX, testY, coherent_field_slice, \
                                    gblur=True, sigma_blur=sigma_blur, show=False)

if show_naive_waist:
    beam_waist_outline = plt.Circle((0,0), w0*1e6, fc='none', ec='r', lw=3, ls='--')
    axarr[0].add_patch(beam_waist_outline)

# fig2, ax = beams.plot_marginalized_profile(testX, testY, field[:,:,focal_plane_z_index], lw=3, \
#                                            axis=0, gblur=True, sigma_blur=sigma_blur, show=False)
fig2, ax = beams.plot_marginalized_profile(testX, testY, field0[:,:,i], \
                                           axis=0, gblur=True, sigma_blur=sigma_blur, show=False, lw=1)
fig2, ax = beams.plot_marginalized_profile(testX, testY, incoherent_field_slice, \
                                           axis=0, gblur=True, sigma_blur=sigma_blur, show=False,\
                                           fig=fig2, ax=ax, lw=3)
fig2, ax = beams.plot_marginalized_profile(testX, testY, incoherent_field_slice2, \
                                           axis=0, gblur=True, sigma_blur=sigma_blur, show=True,\
                                           fig=fig2, ax=ax, lw=3)

# plt.show()
