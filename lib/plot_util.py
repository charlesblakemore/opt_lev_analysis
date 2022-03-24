import sys, os, re

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib.ticker import Locator
from mpl_toolkits.axes_grid1 import make_axes_locatable



def get_single_color(val, cmap='plasma', vmin=0.0, vmax=1.0, log=False):
    '''Gets a single color from a colormap. Useful when the values
       span a continuous range with uneven spacing.

        INPUTS:

            val - value between vmin and vmax, which represent
              the ends of the colormap

            cmap - color map for final output

            vmin - minimum value for the colormap

            vmax - maximum value for the colormap

        OUTPUTS: 

           color - single color in rgba format
    '''

    if (val > vmax) or (val < vmin):
        raise ValueError("Input value doesn't conform to limits")

    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)


    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    return cmap(norm(val))



def get_color_map( n, cmap='plasma', log=False, invert=False):
    '''Gets a map of n colors from cold to hot for use in
       plotting many curves.
       
        INPUTS: 

            n - length of color array to make
            
            cmap - color map for final output

            invert - option to invert

        OUTPUTS: 

            outmap - color map in rgba format
    '''

    n = int(n)
    outmap = []

    if log:
        cNorm = colors.LogNorm(vmin=0, vmax=2*n)
    else:
        cNorm = colors.Normalize(vmin=0, vmax=2*n)

    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for i in range(n):
        outmap.append( scalarMap.to_rgba(2*i + 1) )

    if invert:
        outmap = outmap[::-1]

    return outmap



def truncate_colormap(cmap, vmax=0.0, vmin=1.0, n=256):

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax),
        cmap(np.linspace(vmin, vmax, n)))
    return new_cmap




def add_colorbar(fig, ax, size=0.1, pad=0.05, vmin=0.0, vmax=1.0, \
                 log=False, cmap='plasma', position='right', label='', \
                 labelpad=5, fontsize=14):
    '''
    Adds a colorbar to an existing plot. I find myself rewriting this
    particular set of commands a lot so why not have a function.

    INPUTS:

        fig - pyplot figure instance to be modified

        ax - pyplot axes instance on which to append colorbar

        size - float in [0, 1.0], width of color bar in units of
            full figure width/height (depending on position). if 
            'position' option is right, this number represents 
            the width of the colorbar relative to the full figure width

        vmin - float, min value associated to the colorbar

        vmax - float, max value associated to the colorbar

        log - boolean, specifies logarithmic color spacing

        cmap - str or matplotlib colormap instance, colormap for the
            colorbar

        position - str, 'left', 'right', 'top', 'bottom', spine
            of the axes object on which the colorbar is appended

        label - str, label for the colorbar

        labelpad - float, padding between colorbar ticklabels and colorbar
            axis label, in units of pts

        fontsize - float, font size for colorbar label


    OUTPUTS:

        fig - the modified pyplot figure instance

        ax - the modified axes instance

    '''

    if position in ['left', 'right']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'
    else:
        raise ValueError("'position' argument should be 'left'"\
                            + ", 'right', 'bottom', or 'top'")

    if type(cmap) == str:
        cmap = cm.get_cmap(cmap)

    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)

    ax_cb = divider.append_axes(position, size, pad=pad)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, \
                                   orientation=orientation)
    cb.set_label(label, labelpad=labelpad)

    fig.add_axes(ax_cb)

    return ax_cb, cb








 
########### EXAMPLE USAGE ############       
# ax.set_yscale('symlog', linthreshy=linthresh)
# ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh))


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))




