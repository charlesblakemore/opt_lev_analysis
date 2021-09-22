import os, math
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

import matplotlib.patheffects as patheffects
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines

# matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.size': 20})


limit_ax = 2
limit_path = '/data/new_trap_processed/limits/20200320_limit_binning-10000.p'

plot_projection = True
projection_paths = [ \
                    '/data/new_trap_processed/limits/20200320_noise-limit_discovery.p', \
                    # '/data/new_trap_processed/limits/20200320_noise-limit_discovery_closer.p', \
                    '/data/new_trap_processed/limits/20200320_noise-limit_discovery_closer_scaled.p' \
                   ]
projection_facs = [ \
                   np.sqrt(10000.0 / 100000.0), \
                   np.sqrt(10000.0 / (30.0 * 24.0 * 3600.0)), \
                  ]
# projection_labels = [ \
#                      '$\\Delta x = 14~\\mu$, $\\Delta z = -15~\\mu$m, $T=10^{5}$ s\n$\\sigma = 6 \\times 10^{-17} ~ \\rm{N/rt(Hz)}$', \
#                      '$\\Delta x = 7.5~\\mu$, $\\Delta z = -5~\\mu$m, $T=1$ month\n$\\sigma = 1 \\times 10^{-18} ~ \\rm{N/rt(Hz)}$' \
#                     ]
projection_labels = [ \
                     'Present noise limit', \
                     'Projected next run' \
                    ]
projection_linestyles = [ \
                         '-.', \
                         (0, (3, 1.5, 1, 1.5, 1, 1.5)) \
                        ]
projection_colors = [ \
                     (0.973, 0.586, 0.252, 1.0), \
                     (0.495, 0.012, 0.658, 1.0) \
                    ]


limit_data = pickle.load( open(limit_path, 'rb') )

lambda_vals = limit_data['pos_limit_by_axis'][0]
avg_limit = np.mean( np.stack([limit_data['pos_limit_by_axis'][limit_ax+1], \
                               limit_data['neg_limit_by_axis'][limit_ax+1]], \
                               axis=0), axis=0)

out_arr = np.stack([lambda_vals, avg_limit], axis=0)
np.savetxt('./prev_meas/blakemore_prd_104_l061101_2021.txt', out_arr.T, delimiter=', ')




# if plot_limit:
#     limit_data = pickle.load( open(limit_path, 'rb') )
#     if signed_limit:
#         ax.loglog(limit_data['pos_limit_by_axis'][0]*1e6, \
#                   limit_data['pos_limit_by_axis'][limit_ax+1], \
#                   label='$\\hat{\\alpha} > 0$', color='r', lw=2, ls='--', zorder=7)