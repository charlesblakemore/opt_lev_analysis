import sys, time, itertools

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti
import scipy.linalg as linalg

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

import warnings
warnings.filterwarnings("ignore")





# Various fitting functions
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def line(x, a, b):
    return a * x + b

def plane(x, z, a, b, c):
    return a * x + b * z + c

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
    '''

    # Load modified gravity curves from simulation output
    Gdata = np.load(theory_data_dir + 'Gravdata.npy')
    yukdata = np.load(theory_data_dir + 'yukdata.npy')
    lambdas = np.load(theory_data_dir + 'lambdas.npy')
    xpos = np.load(theory_data_dir + 'xpos.npy')
    ypos = np.load(theory_data_dir + 'ypos.npy')
    zpos = np.load(theory_data_dir + 'zpos.npy')
    
    if lambdas[-1] > lambdas[0]:
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








def get_data_at_harms(files, minsep=20, maxthrow=80, beadheight=5,\
                      cantind=0, ax1='x', ax2='z', diag=True, plottf=False, \
                      tfdate='', tophatf=1000, width=0, nharmonics=10, harms=[], \
                      ext_cant_drive=False, ext_cant_ind=1, plotfilt=False, \
                      max_file_per_pos=1000, userlims=[], noiseband=10):
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

    #parts = data_dir.split('/')
    #prefix = parts[-1]
    #savepath = '/processed_data/grav_data/' + prefix + '_fildat.p'
    #try:
    #    fildat = pickle.load(open(savepath, 'rb'))
    #    return fildat
    #except:
    #    print 'Loading data from: ', data_dir

    fildat = {}
    temp_gdat = {}
    for fil_ind, fil in enumerate(files):
        bu.progress_bar(fil_ind, len(files), suffix=' Sorting Files, Extracting Data')

        ### Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
    
        cantbias = df.electrode_settings['dc_settings'][0]
        ax1pos = df.stage_settings[ax1 + ' DC']
        ax2pos = df.stage_settings[ax2 + ' DC']


        ### Transform cantilever coordinates to bead-centric 
        ### coordinates
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

        if len(userlims):
            if (newxpos < userlims[0][0]*1e6) or (newxpos > userlims[0][1]*1e6):
                #print 'skipped x'
                continue

            if (newheight < userlims[2][0]*1e6) or (newheight > userlims[2][1]*1e6):
                #print 'skipped z'
                continue

        ### Add this combination of positions to the output
        ### data dictionary
        if cantbias not in fildat.keys():
            fildat[cantbias] = {}
        if ax1pos not in fildat[cantbias].keys():
            fildat[cantbias][ax1pos] = {}
        if ax2pos not in fildat[cantbias][ax1pos].keys():
            fildat[cantbias][ax1pos][ax2pos] = []

        if len(fildat[cantbias][ax1pos][ax2pos]) >= max_file_per_pos:
            continue

        if fil_ind == 0 and plottf:
            df.diagonalize(date=tfdate, maxfreq=tophatf, plot=True)
        else:
            df.diagonalize(date=tfdate, maxfreq=tophatf)


        #if fil_ind == 0:
        ginds, fund_ind, drive_freq, drive_ind = \
                df.get_boolean_cantfilt(ext_cant_drive=ext_cant_drive, ext_cant_ind=ext_cant_ind, \
                                        nharmonics=nharmonics, harms=harms, width=width)

        datffts, diagdatffts, daterrs, diagdaterrs = \
                    df.get_datffts_and_errs(ginds, drive_freq, noiseband=noiseband, plot=plotfilt, \
                                            diag=diag)

        df.get_force_v_pos()
        binned = np.array(df.binned_data)
        for resp in [0,1,2]:
            binned[resp][1] = binned[resp][1] * df.conv_facs[resp]

        drivevec = df.cant_data[drive_ind]
        
        mindrive = np.min(drivevec)
        maxdrive = np.max(drivevec)

        posvec = np.linspace(mindrive, maxdrive, 500)
        ones = np.ones_like(posvec)

        pts = np.stack((newxpos*ones, posvec, newheight*ones), axis=-1)

        fildat[cantbias][ax1pos][ax2pos].append((drivevec, posvec, pts, ginds, \
                                                 datffts, diagdatffts, daterrs, diagdaterrs, \
                                                 binned))

    return fildat







def save_fildat(outname, fildat):
    pickle.dump(fildat, open(outname, 'wb'))


def load_fildat(filename):
    fildat = pickle.load(open(filename, 'rb'))
    return fildat









def find_alpha_vs_file(fildat, gfuncs, yukfuncs, lambdas, lims, diag=False, \
                       ignoreX=False, ignoreY=False, ignoreZ=False, plot_best_alpha=False):
    '''Loops over the output from get_data_at_harms, fits each set of
       data against the appropriate modified gravity template and compiles
       the result into a dictionary

       INPUTS: 

       OUTPUTS: 
    '''

    outdat = {}
    temp_gdat = {}
    plot_forces = {}

    biasvec = fildat.keys()
    ax1vec = fildat[biasvec[0]].keys()
    ax2vec = fildat[biasvec[0]][ax1vec[0]].keys()

    for bias in biasvec:
        outdat[bias] = {}
        for ax1 in ax1vec:
            outdat[bias][ax1] = {}
            temp_gdat[ax1] = {}
            plot_forces[ax1] = {}
            for ax2 in ax2vec:
                outdat[bias][ax1][ax2] = []
                temp_gdat[ax1][ax2] = [[],[]]
                temp_gdat[ax1][ax2][1] = [[] for i in range(len(lambdas))]
                plot_forces[ax1][ax2] = [[],[]]
                plot_forces[ax1][ax2][1] = [[] for i in range(len(lambdas))]


    i = 0
    totlen = len(biasvec) * len(ax1vec) * len(ax2vec)
    for bias, ax1, ax2 in itertools.product(biasvec, ax1vec, ax2vec):
        i += 1
        suff = '%i / %i param combinations' % (i, totlen)
        newline=False
        if i == totlen:
            newline=True

        dat = fildat[bias][ax1][ax2]

        nfiles = len(dat)
        filfac = 1.0 / float(nfiles)


        drivevec_avg = np.zeros_like(dat[0][0])
        posvec_avg = np.zeros_like(dat[0][1])
        pts_avg = np.zeros_like(dat[0][2])

        old_ginds = []

        datfft_avg = np.zeros_like(dat[0][4])
        diagdatfft_avg = np.zeros_like(dat[0][5])
        daterr_avg = np.zeros_like(dat[0][6])
        diagdaterr_avg = np.zeros_like(dat[0][7])
        binned_avg = np.zeros_like(dat[0][8])

        for fil in dat:
            drivevec, posvec, pts, ginds, datfft, diagdatfft, daterr, diagdaterr, binned_data = fil
            if not len(old_ginds):
                old_ginds = ginds
            np.testing.assert_array_equal(ginds, old_ginds, err_msg="notch filter inds don't match")
            old_ginds = ginds

            drivevec_avg += filfac * drivevec 
            posvec_avg += filfac * posvec
            pts_avg += filfac * pts
            datfft_avg += filfac * datfft
            diagdatfft_avg += filfac * diagdatfft
            daterr_avg += filfac * daterr
            diagdaterr_avg += filfac * diagdaterr
            binned_avg += filfac * binned_data


        best_fit_alphas = np.zeros(len(lambdas))
        best_fit_errs = np.zeros(len(lambdas))
        diag_best_fit_alphas = np.zeros(len(lambdas))

        for lambind, yuklambda in enumerate(lambdas):
            bu.progress_bar(lambind, len(lambdas), suffix=suff, newline=newline)

            gfft = [[], [], []]
            yukfft = [[], [], []]
            gforce = [[], [], []]
            yukforce = [[], [], []]

            for resp in [0,1,2]:
                if (ignoreX and resp == 0) or (ignoreY and resp == 1) or (ignoreZ and resp == 2):
                    gfft[resp] = np.zeros(np.sum(ginds))
                    yukfft[resp] = np.zeros(np.sum(ginds))
                    continue

                if len(temp_gdat[ax1][ax2][0]):
                    gfft[resp] = temp_gdat[ax1][ax2][0][resp]
                    gforce[resp] = plot_forces[ax1][ax2][0][resp]
                else:
                    gforcevec = gfuncs[resp](pts_avg*1e-6)
                    gforcefunc = interp.interp1d(posvec_avg, gforcevec)
                    gforcet = gforcefunc(drivevec_avg)

                    gforce[resp] = gforcevec
                    gfft[resp] =  np.fft.rfft(gforcet)[ginds]

                if len(temp_gdat[ax1][ax2][1][lambind]):
                    yukfft[resp] = temp_gdat[ax1][ax2][1][lambind][resp]
                    yukforce[resp] = plot_forces[ax1][ax2][0][resp]
                else:
                    yukforcevec = yukfuncs[resp][lambind](pts_avg*1e-6)
                    yukforcefunc = interp.interp1d(posvec_avg, yukforcevec)
                    yukforcet = yukforcefunc(drivevec_avg)

                    yukforce[resp] = yukforcevec
                    yukfft[resp] = np.fft.rfft(yukforcet)[ginds]

            gfft = np.array(gfft)
            yukfft = np.array(yukfft)
            gforce = np.array(gforce)
            yukforce = np.array(yukforce)

            temp_gdat[ax1][ax2][0] = gfft
            temp_gdat[ax1][ax2][1][lambind] = yukfft
            plot_forces[ax1][ax2][0] = gforce
            plot_forces[ax1][ax2][1][lambind] = yukforce


            newalpha = 2 * np.mean( np.abs(datfft_avg) ) / np.mean( np.abs(yukfft) ) * 10**(-1)
            #print newalpha, ':', 
            testalphas = np.linspace(-1.0*newalpha, newalpha, 21)


            chi_sqs = np.zeros(len(testalphas))
            diagchi_sqs = np.zeros(len(testalphas))

            for alphaind, testalpha in enumerate(testalphas):

                chi_sq = 0
                diagchi_sq = 0
                N = 0

                for resp in [0,1,2]:
                    if (ignoreX and resp == 0) or \
                       (ignoreY and resp == 1) or \
                       (ignoreZ and resp == 2):
                        continue
                    re_diff = datfft_avg[resp].real - \
                              (gfft[resp].real + testalpha * yukfft[resp].real )
                    im_diff = datfft_avg[resp].imag - \
                              (gfft[resp].imag + testalpha * yukfft[resp].imag )
                    if diag:
                        diag_re_diff = diagdatfft_avg[resp].real - \
                                       (gfft[resp].real + testalpha * yukfft[resp].real )
                        diag_im_diff = diagdatfft_avg[resp].imag - \
                                       (gfft[resp].imag + testalpha * yukfft[resp].imag )

                    #plt.plot(np.abs(re_diff))
                    #plt.plot(daterr[resp])
                    #plt.show()

                    chi_sq += ( np.sum( np.abs(re_diff)**2 / (0.5*(daterr_avg[resp]**2)) ) + \
                              np.sum( np.abs(im_diff)**2 / (0.5*(daterr_avg[resp]**2)) ) )
                    if diag:
                        diagchi_sq += ( np.sum( np.abs(diag_re_diff)**2 / \
                                                (0.5*(diagdaterr_avg[resp]**2)) ) + \
                                        np.sum( np.abs(diag_im_diff)**2 / \
                                                (0.5*(diagdaterr_avg[resp]**2)) ) )

                    N += len(re_diff) + len(im_diff)

                chi_sqs[alphaind] = chi_sq / (N - 1)
                if diag:
                    diagchi_sqs[alphaind] = diagchi_sq / (N - 1)

            max_chi = np.max(chi_sqs)
            if diag:
                max_diagchi = np.max(diagchi_sqs)

            max_alpha = np.max(testalphas)

            p0 = [max_chi/max_alpha**2, 0, 1]
            if diag:
                diag_p0 = [max_diagchi/max_alpha**2, 0, 1]

            try:
                #plt.plot(testalphas, chi_sqs)
                #plt.show()
                popt, pcov = opti.curve_fit(parabola, testalphas, chi_sqs, \
                                            p0=p0, maxfev=100000)
                if diag:
                    diagpopt, diagpcov = opti.curve_fit(parabola, testalphas, diagchi_sqs, \
                                                        p0=diag_p0, maxfev=1000000)
            except:
                print "Couldn't fit"
                popt = [0,0,0]
                popt[2] = np.mean(chi_sqs)

            best_fit = -0.5 * popt[1] / popt[0]
            fit_err = best_fit * np.sqrt( (pcov[1,1] / popt[1]**2) + (pcov[0,0] / popt[0]**2) )

            best_fit_alphas[lambind] = best_fit
            best_fit_errs[lambind] = fit_err
            if diag:
                diag_best_fit_alphas[lambind] = -0.5 * diagpopt[1] / diagpopt[0]

            stop = time.time()
            #print 'func eval time: ', stop-start

            if plot_best_alpha:

                fig_best, axarr_best = plt.subplots(3,1)
                for resp in [0,1,2]:
                    yforce = (yukforce[resp]-np.mean(yukforce[resp])) * best_fit_alphas[lambind]
                    axarr_best[resp].plot(posvec_avg, yforce, color='r')
                    axarr_best[resp].plot(binned_avg[resp][0], binned_avg[resp][1], color='k')
                plt.show()

        outdat[bias][ax1][ax2] = [best_fit_alphas, best_fit_errs, diag_best_fit_alphas]

    return outdat






def save_alphadat(outname, alphadat, lambdas, minsep, maxthrow, beadheight):
    dump = {}
    dump['alphadat'] = alphadat
    dump['lambdas'] = lambdas
    dump['minsep'] = minsep
    dump['maxthrow'] = maxthrow
    dump['beadheight'] = beadheight

    pickle.dump(dump, open(outname, 'wb'))


def load_alphadat(filename):
    stuff = pickle.load(open(filename, 'rb'))
    return stuff






def fit_alpha_vs_alldim(alphadat, lambdas, minsep=10.0, maxthrow=80.0, beadheight=40.0, \
                        plot=False, scale_fac=1.0*10**9):

    biasvec = alphadat.keys()
    ax1vec = alphadat[biasvec[0]].keys()
    ax2vec = alphadat[biasvec[0]][ax1vec[0]].keys()

    ### Assume separations are encoded in ax1 and heights in ax2
    seps = maxthrow + minsep - np.array(ax1vec)
    heights = np.array(ax2vec)- beadheight

    sort1 = np.argsort(seps)
    sort2 = np.argsort(heights)

    seps_sort = seps[sort1]
    heights_sort = heights[sort2]
    
    heights_g, seps_g = np.meshgrid(heights_sort, seps_sort)

    fits = {}
    outdat = {}

    testvec = []

    ### Perform identical fits for each cantilever bias and height
    for bias in biasvec:
        fits[bias] = {}
        outdat[bias] = {}

        for lambind, yuklambda in enumerate(lambdas):

            dat = [[[] for i in range(len(heights))] for j in range(len(seps))]
            errs = [[[] for i in range(len(heights))] for j in range(len(seps))]

            for ax1ind, ax1pos in enumerate(ax1vec):

                ### Loop over all files at each separation and collect
                ### the value of alpha for the current value of yuklambda
                for ax2ind, ax2pos in enumerate(ax2vec):

                    tempdat = alphadat[bias][ax1pos][ax2pos]

                    dat[ax1ind][ax2ind] = tempdat[0][lambind]
                    errs[ax1ind][ax2ind] = tempdat[1][lambind]

            dat = np.array(dat)
            errs = np.array(errs)

            dat = dat[sort1,:]
            dat = dat[:,sort2]

            errs = errs[sort1,:]
            errs = errs[:,sort2]


            dat_sc = dat * (1.0 / scale_fac)
            errs_sc = errs * (1.0 / scale_fac)


            def func(params, fdat=dat_sc, ferrs=errs_sc):
                funcval = params[0] * heights_g + params[1] * seps_g + params[2]
                return ((funcval - fdat) / ferrs).flatten()

            res = opti.leastsq(func, [0.2*np.mean(dat_sc), 0.2*np.mean(dat_sc), 0], \
                               full_output=1, maxfev=10000)

            try:
                x = res[0]
                residue = linalg.inv(res[1])[2,2]
            except:
                2+2


            deplaned = dat - scale_fac * (x[0] * heights_g + x[1] * seps_g + x[2])
            deplaned_avg = np.mean(deplaned)
            deplaned_std = np.std(deplaned)

            if plot:
                major_ticks = np.arange(15, 21, 1)

                vals = x[0] * heights_g + x[1] * seps_g + x[2]
                vals = vals * scale_fac

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(heights_g, seps_g, vals, rstride=1, cstride=1, alpha=0.3, \
                                color='r')
                ax.scatter(heights_g, seps_g, dat, label='Best-Fit Alphas')
                ax.legend()
                ax.set_xlabel('Z-position [um]')
                ax.set_ylabel('X-separation [um]')
                ax.set_yticks(major_ticks)
                ax.set_zlabel('Alpha [arb]')
                plt.show()
            

            fits[bias][lambind] = (x, residue)
            outdat[bias][lambind] = (dat, errs)

            testvec.append([np.abs(x[2]*scale_fac), deplaned_avg, deplaned_std])

    testvec = np.array(testvec)
    testvec.shape

    alphas_1 = np.array(testvec[:,0])
    alphas_2 = np.array(testvec[:,1])
    alphas_3 = np.array(testvec[:,2])

    return fits, outdat, alphas_1, alphas_2, alphas_3



