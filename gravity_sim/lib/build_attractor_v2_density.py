import time, sys

import numpy as np
import matplotlib.pyplot as plt


### Keep all the parameters in a dictionary so they can be easily
### saved along with the simulation output
attractor_params = {}

### Material densities in kg/m^3
attractor_params['rho_gold'] = 19300.0
attractor_params['rho_silicon'] = 2532.59


### Properties of the attractor, SI (mks) units
### for now this is assumed to be an odd integer,
attractor_params['n_goldfinger'] = 9

attractor_params['width_goldfinger'] = 25.0e-6
attractor_params['width_siliconfinger'] = 25.0e-6
attractor_params['width_outersilicon'] = 25.0e-6

attractor_params['height'] = 10.0e-6

attractor_params['finger_length'] = 80.0e-6

attractor_params['silicon_bridge'] = 2.0e-6
attractor_params['include_bridge'] = True



##################################################################
##################################################################
##################################################################
##################################################################



attractor_params['total_width'] = 
                attractor_params['n_goldfinger'] \
                        * attractor_params['width_goldfinger'] \
              + (attractor_params['n_goldfinger'] - 1) \
                        * attractor_params['width_siliconfinger'] \
              + 2.0 * attractor_params['width_outersilicon']


def density_symmetric(x, y, z):
    '''Function to return the density at the desired point (x,y,z)
       based on the above defined properties of the attractor.
       Coordinate system has it's origin at the center of the front
       face of the attractor, with positive x extending away from the 
       bulk of the device. It is explicitly NOT well-defined at the 
       edges, so the user should be careful to sample points such that 
       they represent centers of rectangular volume elements with size
       given by dx x dy x dz. Currently, edges are exclusive, such 
       that the total mass of the attractor would be under-estimated,
       rather than over-estimated.'''

    ### Establish the easy cases first, returning the appropriate value
    ### and hopefully decreasing runtime
    h = attractor_params['height']
    wtot = attractor_params['total_width']
    within_x = x < 0
    within_z = np.abs(z) < 0.5 * h
    within_y = np.abs(y) < 0.5 * wtot
    if not (within_x and within_y and within_z):
        return 0.0

    rhog = attractor_params['rho_gold']
    rhos = attractor_params['rho_silicon']

    wg = attractor_params['width_goldfinger']
    ws = attractor_params['width_siliconfinger']
    wo = attractor_params['width_outersilicon']
    l = attractor_params['finger_length']
    b = attractor_params['silicon_bridge']
    b_bool = attractor_params['include_bridge']

    ### Boolean flags for various locations 
    central_finger = (np.abs(y) < 0.5 * wg)
    back_bar = ( np.abs(x) < (l + b*b_bool + wg)) \
                and ( np.abs(x) > (l + b*b_bool))
    bridge = b_bool and (x < 0) and (np.abs(x) < b)
    outer = (np.abs(y) > (0.5 * wtot - wo))
    silicon_bulk = np.abs(x) > (l + wg + b*b_bool)

    #### Return the right density value baed on the boolean flags
    if (central_finger and not bridge) or (back_bar and not outer):
        return rhog
    elif bridge or outer or silicon_bulk:
        return rhos

    ### Hardest cases, assuming above aren't true. Naturally assumes a 
    ### symmetric attractor with a gold finger in the center, but that
    ### should be reasonably easy to change
    extra_y = np.abs(y) - 0.5 * wg   ### assumption that n_finger is odd
    extra_y_unit = extra_y % (wg + ws)
    if extra_y_unit > ws:
        return rhog
    elif extra_y_unit < ws:
        return rhos
    else:
        ### Failure case, so the function at least returns a number
        print('No valid condition met, check some shit')
        return 0.0



def build_3d_array(x_range=(-199.5e-6, 0e-6), dx=1.0e-6, \
                    y_range=(-249.5e-6, 250e-6), dy=1.0e-6, \
                    z_range=(-4.5e-6, 5e-6), dz=1.0e-6, \
                    verbose=False):
    '''Build a 3D array of unit cells with coordinates defining
       the center of each unit cell. Grid values are the densities
       of those unit cells, assumed to be entirely one material.

       Default arguments setup 1 um unit cells, spanning most of the
       attractor and a little empty space to either side. Endpoints
       in the {x,y,z}_range arguments need to be chosen carefully.'''

    start = time.time() ### A timer

    ### Build the x, y, and z arrays, and a zero array for the output
    xx = np.arange(x_range[0], x_range[1], dx)
    yy = np.arange(y_range[0], y_range[1], dy)
    zz = np.arange(z_range[0], z_range[1], dz)
    rho_grid = np.zeros((len(xx), len(yy), len(zz)))

    ### Shitty for loop over the values chosen
    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            for k, z in enumerate(zz):
                rho_grid[i,j,k] = density_symmetric(x,y,z)
    stop = time.time() ### stop the timer
    deltat = stop - start

    if verbose:
        print()
        print('With {:d} xpts, {:d} ypts, and {:d} zpts: {:0.5f} second runtime'\
                .format(len(xx), len(yy), len(zz), deltat))

    return xx, yy, zz, rho_grid




def plot_xy_density(zpos=0.0, x_range=(-599.5e-6, 10.0e-6), dx=1.0e-6, \
                    y_range=(-249.5e-6, 250e-6), dy=1.0e-6, \
                    verbose=False, cmap='plasma'):
    '''Plotting function for qualitative validation of the attractor density function
       subject to the same edge effect issues discussed above.'''

    start = time.time()

    ### Build the x and y arrays for plotting
    xx = np.arange(x_range[0], x_range[1], dx)
    yy = np.arange(y_range[0], y_range[1], dy)
    rho_grid = np.zeros((xpts, ypts))

    ### Shitty slow for loops. The density function is hard to vectorize
    ### given the somewhat complex structure of the attractor
    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            rho_grid[i,j] = density_symmetric(x, y, zpos)

    ### Some timing
    stop = time.time()
    deltat = stop - start

    if verbose:
        print()
        print('plot dx [um] : {:0.2f}'.format(dx*1e6))
        print('plot dy [um] : {:0.2f}'.format(dy*1e6))
        print('With {:d} xpts and {:d} ypts: {:0.5f} second runtime'\
                .format(len(xx), len(yy), deltat))

    fig, ax = plt.subplots(1,1)
    ### Plot the transpose since arrays when plotted as images are interpreted as 
    ### (row, column) which translates to (y, x), assuming x is horizontal, thus 
    ### requiring the transpose
    img = ax.imshow(rho_grid.T, cmap=cmap, \
                aspect=(x_range[1] - x_range[0])/(y_range[1] - y_range[0]),\
                extent=[x_range[0]*1e6, x_range[1]*1e6, y_range[0]*1e6, y_range[1]*1e6], \
                vmin=0.0, vmax=20000)

    rho_silicon = attractor_params['rho_silicon']
    rho_gold = attractor_params['rho_gold']

    ### Some custom and hardcoded ticks assuming we will be using gold and silicon to 
    ### to make our attractors, might be worthwhile to make this dynamic for two materials
    tick_locs = [0.0, rho_silicon, 5000.0, 7500.0, 10000.0, 12500.0,\
                 15000.0, 17500.0, rho_gold]
    tick_labels = ['0', '{:0.1f} - Si'.format(rho_silicon), '5000', '7500', \
                    '10000', '12500', '15000', '17500', '{:0.1f} - Au'.format(rho_gold)]

    ax.set_xlabel('$x$-coordinate [um]')
    ax.set_ylabel('$y$-coordinate [um]')
    cbar = fig.colorbar(img, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Density [kg/m$^3$]')
    fig.tight_layout()
    plt.show()




### Uncomment these lines if you want to test it

# xx, yy, zz, rho_grid = build_3d_array(verbose=True)

# print()
# print('Array size: {:0.1f} Megabytes'\
#         .format(float(sys.getsizeof(rho_grid)) / 1.0e6))

# plot_xy_density(verbose=True)