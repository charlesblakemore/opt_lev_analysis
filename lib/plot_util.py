import sys, os, re, math

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





#Label line with line2D label data
def labelLine(line, x, x_offset=0.0, y_offset=0.0, \
              alpha=0.0, label=None, align=True, **kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = math.degrees(math.atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5
        
    t = ax.text(x+x_offset, y+y_offset, label, rotation=trans_angle, **kwargs)
    t.set_bbox(dict(alpha=alpha))






 
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




