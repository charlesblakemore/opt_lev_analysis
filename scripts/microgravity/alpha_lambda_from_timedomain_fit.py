import sys, time, itertools

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

import warnings
warnings.filterwarnings("ignore")


##################################################################
######################## Script Params ###########################

only_closest = True
minsep = 20       # um
maxthrow = 80     # um
beadheight = 11   # um


data_dir = '/data/20180308/bead2/grav_data/init'
savepath = '/sensitivities/20180308_grav_init_close_time.npy'
save = False
load = False

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

tfdate = '' #'20180215'

confidence_level = 0.95

#user_lims = [(65e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
user_lims = [(5e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
#user_lims = []

tophatf = 300   # Hz, doesn't reconstruct data above this frequency

plot_just_current = False
figtitle = ''

print_timing = False


##################################################################
################# Constraints to plot against ####################

#limitdata_path = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
#                 'limitdata_20160928_datathief_nodecca2.txt'
limitdata_path = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
                 'decca2_limit.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'


#limitdata_path2 = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
#                  'limitdata_20160914_datathief.txt'
limitdata_path2 = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
                 'no_decca2_limit.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'









##################################################################
##################################################################
##################################################################


# Various fitting functions
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def line(x, a, b):
    return a * x + b

def const(x, a):
    return a

def flicker(x, a):
    return a * (1. / x)


def build_mod_grav_funcs(theory_data_dir):
    '''Loads data from the output of /data/grav_sim_data/process_data.py
       which processes the raw simulation output from the farmshare code

       INPUTS: theory_data_dir, path to the directory containing the data

       OUTPUTS: gfuncs, 3 element list with 3D interpolating functions
                        for regular gravity [fx, fy, fz]
                yukfuncs, 3 x Nlambda array with 3D interpolating function
                          for modified gravity with indexing: 
                          [[y0_fx, y1_fx, ...], [y0_fy, ...], [y0_fz, ...]]
                lambdas, np.array with all lambdas from the simulation
                lims, 3 element with tuples for (min, max) of coordinate
                      limits in interpolation
    '''

    # Load modified gravity curves from simulation output
    Gdata = np.load(theory_data_dir + 'Gravdata.npy')
    yukdata = np.load(theory_data_dir + 'yukdata.npy')
    lambdas = np.load(theory_data_dir + 'lambdas.npy')
    xpos = np.load(theory_data_dir + 'xpos.npy')
    ypos = np.load(theory_data_dir + 'ypos.npy')
    zpos = np.load(theory_data_dir + 'zpos.npy')
    
    lambdas = lambdas[::-1]
    yukdata = np.flip(yukdata, 0)

    # Find limits to avoid out of range erros in interpolation
    xlim = (np.min(xpos), np.max(xpos))
    ylim = (np.min(ypos), np.max(ypos))
    zlim = (np.min(zpos), np.max(zpos))

    # Build interpolating functions for regular gravity
    gfuncs = [0,0,0]
    for resp in [0,1,2]:
        gfuncs[resp] = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,resp])

    # Build interpolating functions for yukawa-modified gravity
    yukfuncs = [[],[],[]]
    for resp in [0,1,2]:
        for lambind, yuklambda in enumerate(lambdas):
            lamb_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,resp])
            yukfuncs[resp].append(lamb_func)
    lims = [xlim, ylim, zlim]

    return gfuncs, yukfuncs, lambdas, lims




def get_alpha_lambda(gfuncs, yukfuncs, lambdas, lims, files, cantind=0, \
                     ax1='x', ax2='z', spacing=1e-6, diag=True, plottf=False, \
                     minsep=20, maxthrow=80, beadheight=5, width=0, nharmonics=10, \
                     ignoreX=False, ignoreY=False, ignoreZ=False, \
                     plot=True, save=False, savepath='', confidence_level=0.95,
                     print_timing=False, noiseband=10, only_closest=False):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then performs an optimal filter using the cantilever drive and 
       a theoretical force vs position to generate the filter/template.
       The result of the optimal filtering is stored, and the data 
       released from memory

       INPUTS: files, list of files names to extract data
               cantind, cantilever electrode index
               ax1, axis with different DC positions
               ax2, 2nd axis with different DC positions

       OUTPUTS: 
    '''

    # For the confidence interval, compute the inverse CDF of a 
    # chi^2 distribution at given confidence level and compare to 
    # liklihood ratio via a goodness of fit parameter.
    # Refer to scipy.stats documentation to understand chi2
    chi2dist = stats.chi2(1)
    # factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
    con_val = 0.5 * chi2dist.ppf(confidence_level)

    colors = bu.get_color_map(len(lambdas))

    alphas = np.zeros_like(lambdas)
    
    #testalphas = np.linspace(0, 16, 50)
    testalphas = np.linspace(10**0, 10**14, 50)

    fildat = {}
    for fil_ind, fil in enumerate(files):
        bu.progress_bar(fil_ind, len(files), suffix=' Sorting Files, Extracting Data')

        ### Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
    
        cantbias = df.electrode_settings['dc_settings'][0]
        ax1pos = df.stage_settings[ax1 + ' DC']
        ax2pos = df.stage_settings[ax2 + ' DC']
        
        if cantbias not in fildat.keys():
            fildat[cantbias] = {}
        if ax1pos not in fildat[cantbias].keys():
            fildat[cantbias][ax1pos] = {}
        if ax2pos not in fildat[cantbias][ax1pos].keys():
            fildat[cantbias][ax1pos][ax2pos] = [[], 0, [], []]
            empty = True

        # fildat[cantbias][ax1pos][ax2pos].append(fil)

        if diag:
            if fil_ind == 0 and plottf:
                df.diagonalize(date=tfdate, maxfreq=tophatf, plot=True)
            else:
                df.diagonalize(date=tfdate, maxfreq=tophatf)
                    
        ### Find which axes were driven. If multiple are found,
        ### it takes the axis with the largest amplitude
        indmap = {0: 'x', 1: 'y', 2: 'z'}
        driven = [0,0,0]
        for ind, key in enumerate(['x driven','y driven','z driven']):
            if df.stage_settings[key]:
                driven[ind] = 1
        if np.sum(driven) > 1:
            amp = [0,0,0]
            for ind, val in enumerate(driven):
                if val: 
                    key = indmap[ind] + ' amp'
                    amp[ind] = df.stage_settings[key]
            drive_ind = np.argmax(np.abs(amp))
        else:
            drive_ind = np.argmax(np.abs(driven))

        drivevec = df.cant_data[drive_ind]

        mindrive = np.min(drivevec)
        maxdrive = np.max(drivevec)

        drivefft = np.fft.rfft(drivevec)
        freqs = np.fft.rfftfreq(len(drivevec), d=1.0/df.fsamp)

        # Find the drive frequency, ignoring the DC bin
        fund_ind = np.argmax( np.abs(drivefft[1:]) ) + 1
        drive_freq = freqs[fund_ind]

        drivefilt = np.zeros(len(drivefft))
        drivefilt[fund_ind] = 1.0

        if width:
            lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
            upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
            drivefilt[lower_ind:upper_ind+1] = drivefilt[fund_ind]

        # Generate an array of harmonics
        harms = np.array([x+2 for x in range(nharmonics)])

        # Loop over harmonics and add them to the filter
        for n in harms:
            harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
            drivefilt[harm_ind] = 1.0 
            if width:
                h_lower_ind = harm_ind - (fund_ind - lower_ind)
                h_upper_ind = harm_ind + (upper_ind - fund_ind)
                drivefilt[h_lower_ind:h_upper_ind+1] = drivefilt[harm_ind]

        # Apply filter by indexing with a boolean array
        ginds = drivefilt > 0
        
        if empty:
            freqdat = (freqs, ginds)
        else:
            assert np.sum(freqs[ginds] - freqdat[0][freqdat[1]]) == 0

        datffts = [[], [], []]
        diagdatffts = [[], [], []]

        daterrs = [[], [], []]
        diagdaterrs = [[], [], []]

        for resp in [0,1,2]:
            if ignoreX and resp == 0:
                continue
            if ignoreY and resp == 1:
                continue
            if ignoreZ and resp == 2:
                continue

            datfft = np.fft.rfft(df.pos_data[resp]*df.conv_facs[resp])
            datffts[resp] = datfft[ginds]
            if diag:
                diagdatfft = np.fft.rfft(df.diag_pos_data[resp])
                diagdatffts[resp] = diagdatfft[ginds]

            noise_inds_all = np.abs(freqs - drive_freq) < 0.5*noiseband
            noise_inds = noise_inds_all * ginds

            daterrs[resp] = np.ones_like(datffts[resp]) * np.mean(datfft[noise_inds])
            diagdaterrs[resp] = np.ones_like(diagdatffts[resp]) * np.mean(diagdatfft[noise_inds])

        datffts = np.array(datffts)
        diagdatffts = np.array(diagdatffts)
                
        daterrs = np.array(daterrs)
        diagdaterrs = np.array(diagdaterrs)


        if empty:
            newdat = datffts
            newdiagdat = diagdatffts

            newerr = daterrs
            newdiagerr = diagdaterrs

            newdrive = drivevec

        else:
            olddat, olddiagdat, olderr, olddiagerr = fildat[cantbias][ax1pos][ax2pos][0]
        
            newdat = olddat + datffts
            newdiagdat = olddiagdat + diagdatffts

            newerr = olderr + daterrs
            newdiagerr = olddiagerr + diagdaterrs

            newdrive = newdrive + drivevec

        fildat[cantbias][ax1pos][ax2pos][0] = (newdat, newdiagdat, newerr, newdiagerr)
        fildat[cantbias][ax1pos][ax2pos][1] += 1
        fildat[cantbias][ax1pos][ax2pos][2] = freqdat
        fildat[cantbias][ax1pos][ax2pos][3] = newdrive


    biasvec = fildat.keys()
    biasvec.sort()
    ax1posvec = fildat[biasvec[0]].keys()
    ax1posvec.sort()
    ax2posvec = fildat[biasvec[0]][ax1posvec[0]].keys()
    ax2posvec.sort()

    if only_closest:
        if ax1 == 'x' and ax2 == 'z':
            seps = minsep + (maxthrow - np.array(ax1posvec))
            heights = np.array(ax2posvec) - beadheight
            sind = np.argmin(seps)
            hind = np.argmin(np.abs(heights - beadheight))
            ax1posvec = [ax1posvec[sind]]
            ax2posvec = [ax2posvec[hind]]

        elif ax1 =='z' and ax2 == 'x':
            seps = minsep + (maxthrow - np.array(ax2posvec))
            heights = np.array(ax1pos) - beadheight
            sind = np.argmin(seps)
            hind = np.argmin(np.abs(heights - beadheight))
            ax1posvec = [ax1posvec[hind]]
            ax2posvec = [ax2posvec[sind]]
        


    tot_iterations = len(biasvec) * len(ax1posvec) * len(ax2posvec) * len(lambdas) * len(testalphas)
    i = -1
    
    for lambind, yuklambda in enumerate(lambdas):
        chi_sqs = np.zeros(len(testalphas))
        diagchi_sqs = np.zeros(len(testalphas))

        if print_timing:
            startlamb = time.time()

        for alphaind, testalpha in enumerate(testalphas):
            N = 0
            chi_sq = 0
            diagchi_sq = 0

            if print_timing:
                startalpha = time.time()

            for bias, ax1pos, ax2pos in itertools.product(biasvec, ax1posvec, ax2posvec):
                i += 1
                if not print_timing:
                    bu.progress_bar(i, tot_iterations, suffix=' Fitting the Data for Chi^2')

                ### Setup coodinates: map cantilever positions to positions
                ### relative to bead
                if ax1 == 'x' and ax2 == 'z':
                    newxpos = minsep + (maxthrow - ax1pos)
                    newheight = ax2pos - beadheight
                elif ax1 =='z' and ax2 == 'x':
                    newxpos = minsep + (maxthrow - ax2pos)
                    newheight = ax1pos - beadheight
                else:
                    print "Coordinate axes don't make sense for gravity data..."
                    print "Proceeding anyway, but results might be hard to interpret"
                    newxpos = ax1pos
                    newheight = ax2pos

                if (newxpos < lims[0][0]*1e6) or (newxpos > lims[0][1]*1e6):
                    #print 'skipped x'
                    continue

                if (newheight < lims[2][0]*1e6) or (newheight > lims[2][1]*1e6):
                    #print 'skipped z'
                    continue

                entry = fildat[bias][ax1pos][ax2pos]
    
                datfft, diagdatfft, daterr, diagdaterr = entry[0]
                nffts = entry[1]
                freqs, ginds = entry[2]
                drivevec = entry[3]

                datfft = datfft * (1.0 / nffts)
                diagdatfft = diagdatfft * (1.0 / nffts)

                mindrive = np.min(drivevec)
                maxdrive = np.max(drivevec)

                posvec = np.linspace(mindrive, maxdrive, 500)
                ones = np.ones_like(posvec)
                pts = np.stack((newxpos*ones, posvec, newheight*ones), axis=-1)

                gfft = [[], [], []]
                yukfft = [[], [], []]
                for resp in [0,1,2]:
                    if ignoreX and resp == 0:
                        continue
                    if ignoreY and resp == 1:
                        continue
                    if ignoreZ and resp == 2:
                        continue

                    gforcevec = gfuncs[resp](pts*1e-6)
                    gforcefunc = interp.interp1d(posvec, gforcevec)
                    gforcet = gforcefunc(drivevec)

                    yukforcevec = yukfuncs[resp][lambind](pts*1e-6)
                    yukforcefunc = interp.interp1d(posvec, yukforcevec)
                    yukforcet = yukforcefunc(drivevec)

                    gfft[resp] =  np.fft.rfft(gforcet)[ginds]
                    yukfft[resp] = np.fft.rfft(yukforcet)[ginds]


                gfft = np.array(gfft)
                yukfft = np.array(yukfft)


                for resp in [0,1,2]:

                    diff = datfft[resp] - (gfft[resp] + testalpha * yukfft[resp])
                    if diag:
                        diagdiff = diagdatfft[resp] - \
                                    (gfft[resp] + testalpha * yukfft[resp])

                    chi_sq += np.sum( np.abs(diff)**2 / daterr[resp]**2 )
                    diagchi_sq += np.sum( np.abs(diagdiff)**2 / diagdaterr[resp]**2 )
                    N += len(diff)
    
            red_chi_sq = chi_sq / (N - 1)
            diagred_chi_sq = diagchi_sq / (N - 1)
            chi_sqs[alphaind] = red_chi_sq
            diagchi_sqs[alphaind] = diagred_chi_sq

            if print_timing:
                stopalpha = time.time()
                print "Time per alpha: ", stopalpha - startalpha

        max_chi = np.max(chi_sqs)
        max_alpha = np.max(testalphas)

        p0 = [max_chi/max_alpha**2, 0, 1]

        #if lambind == 0:
        #    p0 = [0.15e9, 0, 5]
        #else:
        #    p0 = p0_old

        if plot:
            plt.plot(testalphas, chi_sqs, color = colors[lambind])
    
        try:
            popt, pcov = opti.curve_fit(parabola, testalphas, chi_sqs, \
                                            p0=p0, maxfev = 100000)
        except:
            print "Couldn't fit"
            popt = [0,0,0]
            popt[2] = np.mean(chi_sqs)

        #p0_old = popt

        con_val = con_val + np.min(chi_sqs)

        # Select the positive root for the non-diagonalized data
        soln1 = ( -1.0 * popt[1] + np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
        soln2 = ( -1.0 * popt[1] - np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
        if soln1 > soln2:
            alpha_con = soln1
        else:
            alpha_con = soln2

        alphas[lambind] = alpha_con
    
        if print_timing:
            stoplamb = time.time()
            print "Time per lambda: ", stoplamb - startlamb


    if plot:
        plt.title('Goodness of Fit for Various Lambda', fontsize=16)
        plt.xlabel('Alpha Parameter [arb]', fontsize=14)
        plt.ylabel('$\chi^2$', fontsize=18)

        plt.show()

    if save:
        if savepath == '':
            print 'No save path given, type full path here'
            savepath = raw_input('path: ')

        np.save(savepath, [lambdas, alphas])


    return lambdas, alphas





if not plot_just_current:
    gfuncs, yukfuncs, lambdas, lims = build_mod_grav_funcs(theory_data_dir)

    datafiles = bu.find_all_fnames(data_dir, ext=config.extensions['data'])

    if len(datafiles) == 0:
        print "Found no files in: ", data_dir
        quit()

    newlambdas, alphas = get_alpha_lambda(gfuncs, yukfuncs, lambdas, lims, datafiles, cantind=0, \
                                          ax1='x', ax2='z', spacing=1e-6, diag=True, plottf=False, \
                                          minsep=minsep, maxthrow=maxthrow, beadheight=beadheight, \
                                          print_timing=print_timing, only_closest=only_closest)


    outdat = [newlambdas, alphas]
    if save:
        np.save(savepath, outdat)

    if load:
        dat = np.load(savepath)
        newlambdas = dat[0]
        alphas = dat[1]






fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

if not plot_just_current:
    ax.loglog(newlambdas, alphas, linewidth=2, label='95% CL')
ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('$\\alpha$')

ax.legend(numpoints=1, fontsize=9)

ax.set_title(figtitle)

plt.tight_layout(w_pad=1.2, h_pad=1.2, pad=1.2)

plt.show()
