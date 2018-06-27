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


confidence_level = 0.95
chi2dist = stats.chi2(1)
# factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
con_val = 0.5 * chi2dist.ppf(confidence_level)



# Various fitting functions
# pretty self explanatory so there are few comments
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


def solve_parabola(yvalue, popt):
    '''Computes the two xvalues corresponding to a given
       yvalue in the equation: y = ax^2 + bx + c.

       INPUTS:   yvalue, solution you're looking for
                 popt, list [a,b,c] paramters of parabola

       OUTPUTS:  soln, tuple with solution (x_small, x_large)
    '''
    a, b, c = popt
    soln1 = (-b + np.sqrt(b**2 - 4.0*a*(c-yvalue))) / (2.0*a)
    soln2 = (-b - np.sqrt(b**2 - 4.0*a*(c-yvalue))) / (2.0*a)
    if soln1 > soln2:
        soln = (soln2, soln1)
    else:
        soln = (soln1, soln2)

    return soln




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

    ### Load modified gravity curves from simulation output
    Gdata = np.load(theory_data_dir + 'Gravdata.npy')
    yukdata = np.load(theory_data_dir + 'yukdata.npy')
    lambdas = np.load(theory_data_dir + 'lambdas.npy')
    xpos = np.load(theory_data_dir + 'xpos.npy')
    ypos = np.load(theory_data_dir + 'ypos.npy')
    zpos = np.load(theory_data_dir + 'zpos.npy')
    
    if lambdas[-1] > lambdas[0]:
        lambdas = lambdas[::-1]
        yukdata = np.flip(yukdata, 0)

    ### Find limits to avoid out of range erros in interpolation
    xlim = (np.min(xpos), np.max(xpos))
    ylim = (np.min(ypos), np.max(ypos))
    zlim = (np.min(zpos), np.max(zpos))

    ### Build interpolating functions for regular gravity
    gfuncs = [0,0,0]
    for resp in [0,1,2]:
        gfuncs[resp] = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,resp])

    ### Build interpolating functions for yukawa-modified gravity
    yukfuncs = [[],[],[]]
    for resp in [0,1,2]:
        for lambind, yuklambda in enumerate(lambdas):
            lamb_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,resp])
            yukfuncs[resp].append(lamb_func)
    lims = [xlim, ylim, zlim]

    return gfuncs, yukfuncs, lambdas, lims








def get_data_at_harms(files, p0_bead=[20,0,20], ax_disc=0.5, \
                      cantind=0, ax1='x', ax2='z', diag=True, plottf=False, \
                      tfdate='', tophatf=1000, width=0, harms=[], nharmonics=10, \
                      ext_cant_drive=False, ext_cant_ind=1, plotfilt=False, \
                      max_file_per_pos=1000, userlims=[], noiseband=10):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then applies a notch filter using the attractor drive. The response
       at the attractor's fundamental + harmonics is returned

       INPUTS: files,      list of files names to extract data
               minsep,     minimum separation between face of cantilever and bead center
               maxthrow,   maximum extension of cantilever (for bootstrapped separation)
               beadheight, height of microspphere in attractor coordinates
               cantind,    cantilever electrode index
               ax1,        string ['x', 'y', 'z'] first axis to index in output
               ax2,        string scond axis to index in output dictionary
               diag,       boolean: use diagonalized data?
               plottf,     boolean: show applied transfer function
               tfdate,     optional string to specify using a tf from a different date
                              leave empty if you want to use the standard tf
               tophatf,    top-hat filter frequency for diagonalization/reconstruction
               width,      width of notch filter in Hz. 0 for single bin notch filter
               harms,      list of harmonics to include in filter. 1 is fundamental
               nharmonics, if harms is empty, the script generates this many harmonics 
                              including the fundamental
               userlims,   limits on the spatial coordinates, outside of which it doesn't
                              analyze the dta
               plotfilt,   boolean to plot the result of the filtering (used to debug)
               ext_cant_drive,  boolean specifying if an external driver was used to position
                                  the attractor. The script uses the auto-generated stage 
                                  parameters otherwise
               ext_cant_ind,     index of the external attractor drive {0: x, 1: y, 2: z}
               max_file_per_pos,  maximum number of files to include per position
                                    if you want to select subsets for statistics

       OUTPUTS:   fildat,  a big ass dictionary with everything you want, and a bunch of 
                             stuff you probably don't want. Has three levels of keys:
                             1st keys (fildat.keys()): cantilever biases
                             2nd keys (fildat[keys1[0]].keys()): ax1 positions
                             3rd keys (fildat[keys1[0]][keys2[0].keys()): ax2 positions

                             each entry for all three keys has a list of tuples, one tuple
                             for each file. Each tuple has the following elements:

                             drivevec, relevant cantilever drive as numpy array
                             posvec,   vector of bead positions relative to origin fixed at 
                                         center of the cantilever's front face
                             pts,      points array for mod_grav_funcs input
                             ginds,    fft frequeny indices of notch filter. if all the data
                                         is syncrhonized, these should be exactly the same 
                                         for all data arrays and files
                             datffts,  averaged value of dataffts at freqs[ginds]
                             diagdatffts, averaged value of diagonalized dataffts at freqs[ginds]
                             daterrs,  errors on averaged datffts at freqs[ginds]
                             diagdaterrs, errors on averaged  diagonalized dataffts at freqs[ginds]
                             binned, force_v_pos curves for plotting and qualitative checks
    '''

    ax_keys = {'x': 0, 'y': 1, 'z': 2}

    fildat = {}
    ax1vec = []
    Nax1 = {}
    ax2vec = []
    Nax2 = {}
    temp_gdat = {}
    for fil_ind, fil in enumerate(files):
        bu.progress_bar(fil_ind, len(files), suffix=' Sorting Files, Extracting Data')

        ### Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
    
        ### Extract the relevant position and bias
        cantbias = df.electrode_settings['dc_settings'][0]

        #ax1pos = df.stage_settings[ax1 + ' DC']
        ax1pos = np.mean(df.cant_data[ax_keys[ax1]])
        ax1pos = round(ax1pos, 1)

        #ax2pos = df.stage_settings[ax2 + ' DC']
        ax2pos = np.mean(df.cant_data[ax_keys[ax2]])
        ax2pos = round(ax2pos, 1)


        ### Add this combination of positions to the output
        ### data dictionary
        if cantbias not in fildat.keys():
            fildat[cantbias] = {}
            Nax1[cantbias] = {}
            Nax2[cantbias] = {}

        ax1_is_new = False
        if len(ax1vec):
            close_ind = np.argmin( np.abs( np.array(ax1vec) - ax1pos ) )
            if np.abs(ax1vec[close_ind] - ax1pos) < ax_disc:
                old_ax1key = ax1vec[close_ind]
                oldN = Nax1[cantbias][old_ax1key]

                new_ax1key = (old_ax1key * oldN + ax1pos) / (oldN + 1.0)
                new_ax1key = round(new_ax1key, 1)

                if old_ax1key != new_ax1key:
                    ax1vec[close_ind] = new_ax1key

                    fildat[cantbias][new_ax1key] = fildat[cantbias][old_ax1key]
                    Nax1[cantbias][new_ax1key] = oldN + 1.0

                    del Nax1[cantbias][old_ax1key]
                    del fildat[cantbias][old_ax1key]
                    
            else:
                ax1_is_new = True
        else:
            ax1_is_new = True

        if ax1_is_new:
            fildat[cantbias][ax1pos] = {}
            Nax1[cantbias][ax1pos] = 1
            new_ax1key = ax1pos
            ax1vec.append(new_ax1key)
            ax1vec.sort()
            

        #ax2keys = fildat[cantbias][new_ax1key].keys()

        ax2_is_new = False
        if len(ax2vec):
            close_ind = np.argmin( np.abs( np.array(ax2vec) - ax2pos ) )
            if np.abs(ax2vec[close_ind] - ax2pos) < ax_disc:

                old_ax2key = ax2vec[close_ind]
                oldN = Nax2[cantbias][old_ax2key]

                new_ax2key = (old_ax2key * oldN + ax2pos) / (oldN + 1.0)
                new_ax2key = round(new_ax2key, 1)

                new_combo = False
                if old_ax2key not in fildat[cantbias][new_ax1key].keys():
                    fildat[cantbias][new_ax1key][new_ax2key] = []

                if old_ax2key != new_ax2key:
                    ax2vec[close_ind] = new_ax2key

                    Nax2[cantbias][new_ax2key] = oldN + 1.0
                    del Nax2[cantbias][old_ax2key]

                    for ax1key in ax1vec:
                        ax2keys = fildat[cantbias][ax1key].keys()
                        if old_ax2key in ax2keys:
                            fildat[cantbias][ax1key][new_ax2key] = fildat[cantbias][ax1key][old_ax2key]
                            del fildat[cantbias][ax1key][old_ax2key]

            else:
                ax2_is_new = True
        else:
            ax2_is_new = True

        if ax2_is_new:
            fildat[cantbias][new_ax1key][ax2pos] = []
            Nax2[cantbias][ax2pos] = 1
            new_ax2key = ax2pos
            ax2vec.append(new_ax2key)
            ax2vec.sort()


        ### Transform cantilever coordinates to bead-centric 
        ### coordinates
        if ax1 == 'x' and ax2 == 'z':
            newxpos = p0_bead[0] + (80 - new_ax1key)
            newheight = new_ax2key - p0_bead[2]
        elif ax1 =='z' and ax2 == 'x':
            newxpos = p0_bead[0] + (80 - new_ax2key)
            newheight = beadheight - new_ax2key 
        else:
            print "Coordinate axes don't make sense for gravity data..."
            print "Proceeding anyway, but results might be hard to interpret"
            newxpos = new_ax1key
            newheight = new_ax2key

        if len(userlims):
            if (newxpos < userlims[0][0]*1e6) or (newxpos > userlims[0][1]*1e6):
                #print 'skipped x'
                continue

            if (newheight < userlims[2][0]*1e6) or (newheight > userlims[2][1]*1e6):
                #print 'skipped z'
                continue


        if len(fildat[cantbias][new_ax1key][new_ax2key]) >= max_file_per_pos:
            continue


        ### Diagonalize the data and plot if desired
        if fil_ind == 0 and plottf:
            df.diagonalize(date=tfdate, maxfreq=tophatf, plot=True)
        else:
            df.diagonalize(date=tfdate, maxfreq=tophatf)


        ### Build the notch filter from the attractor drive
        ginds, fund_ind, drive_freq, drive_ind = \
                df.get_boolean_cantfilt(ext_cant_drive=ext_cant_drive, ext_cant_ind=ext_cant_ind, \
                                        nharmonics=nharmonics, harms=harms, width=width)

        ### Apply notch filter, extract data, and errors
        datffts, diagdatffts, daterrs, diagdaterrs = \
                    df.get_datffts_and_errs(ginds, drive_freq, noiseband=noiseband, plot=plotfilt, \
                                            diag=diag)

        ### Get the binned data and calibrate the non-diagonalized part
        df.get_force_v_pos()
        binned = np.array(df.binned_data)
        for resp in [0,1,2]:
            binned[resp][1] = binned[resp][1] * df.conv_facs[resp]

        ### Analyze the attractor drive and build the relevant position vectors
        ### for the bead
        drivevec = df.cant_data[drive_ind] - np.mean(df.cant_data[drive_ind]) + p0_bead[1]
        mindrive = np.min(drivevec)
        maxdrive = np.max(drivevec)
        posvec = np.linspace(mindrive, maxdrive, 500)
        ones = np.ones_like(posvec)
        pts = np.stack((newxpos*ones, posvec, newheight*ones), axis=-1)

        fildat[cantbias][new_ax1key][new_ax2key].append((drivevec, posvec, pts, ginds, \
                                                         datffts, diagdatffts, daterrs, diagdaterrs, \
                                                         binned))

    return fildat







def save_fildat(outname, fildat):
    '''Pretty self explanatory. It's just a wrapper for a pickling, but if 
       what we save ever gets more complicated, this function can accomodate.'''
    pickle.dump(fildat, open(outname, 'wb'))


def load_fildat(filename):
    '''Same shit as save_fildat().'''
    fildat = pickle.load(open(filename, 'rb'))
    return fildat









def find_alpha_vs_file(fildat, gfuncs, yukfuncs, lambdas, diag=False, \
                       ignoreX=False, ignoreY=False, ignoreZ=False, plot_best_alpha=False):
    '''Loops over the output from get_data_at_harms, fits each set of
       data against the appropriate modified gravity template and compiles
       the result into a dictionary

       INPUTS:  fildat,    object from get_data_at_harms() function above
                gfuncs,    interpolating functions for regular gravity around attractor
                yukfuncs,  interpolating functions for yukawa modified gravity around attractor
                lambdas,   array of lambda values to use
                diag,      boolean to use diagonalized data. not fully realized
                ignoreX,   boolean to ignore X data in least-squared minimization
                ignoreY,   boolean to ignore Y data in least-squared minimization
                ignoreZ,   boolean to ignore Z data in least-squared minimization
                plot_best_alpha,  boolean to plot the best fit alpha and the minimization
                                    that produced that best fit (debug)

       OUTPUTS: alphadat, another big ass dictionary! Has three levels of keys:
                             1st keys (alphadat.keys()): cantilever biases
                             2nd keys (alphadat[keys1[0]].keys()): ax1 positions
                             3rd keys (alphadat[keys1[0]][keys2[0].keys()): ax2 positions

                             each entry for all three keys has a list with the following:
                             best_fit_alphas,  array of best fit alphas with same indexing
                                                 as the lambdas supplied to this function
                             best_fit_errs,    95%CL on best_fit being consistent with 0 
                             diag_best_fit_lphas, self-explanatory
    '''

    outdat = {}
    temp_gdat = {}
    plot_forces = {}

    biasvec = fildat.keys()
    ax1vec = fildat[biasvec[0]].keys()
    ax2vec = fildat[biasvec[0]][ax1vec[0]].keys()

    #if ignoreX and ignoreY and not ignoreZ:
    #    nax2 = len(ax2vec)
    #    mid = int(np.floor(0.5 * nax2))
    #    ax2vec = np.concatenate((ax2vec[:mid-1], ax2vec[mid+2:]))

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


    ### Iterate over all paramter combinations
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

        ### Initialize average arrays 
        drivevec_avg = np.zeros_like(dat[0][0])
        posvec_avg = np.zeros_like(dat[0][1])
        pts_avg = np.zeros_like(dat[0][2])

        old_ginds = []

        datfft_avg = np.zeros_like(dat[0][4])
        diagdatfft_avg = np.zeros_like(dat[0][5])
        daterr_avg = np.zeros_like(dat[0][6])
        diagdaterr_avg = np.zeros_like(dat[0][7])
        binned_avg = np.zeros_like(dat[0][8])

        ### Loop over all files at this combination of parameters and add them
        ### to the averaged array
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

        old_datfft = datfft_avg

        ### Loop over lambda values and fit the averaged response of this combination
        ### of parameters to modified gravity, finding the best-fit value of alpha
        for lambind, yuklambda in enumerate(lambdas):
            bu.progress_bar(lambind, len(lambdas), suffix=suff, newline=newline)

            gfft = [[], [], []]
            yukfft = [[], [], []]
            gforce = [[], [], []]
            yukforce = [[], [], []]

            ### Compute the hypothetical modified gravity response for this attractor-bead
            ### positioning. Convert this F(y) -> F(t) with the known cantilever y(t) by 
            ### by interpolating F vs y data and then sampling said interpolating function
            for resp in [0,1,2]:
                if (ignoreX and resp == 0) or (ignoreY and resp == 1) or (ignoreZ and resp == 2):
                    gfft[resp] = np.zeros(np.sum(ginds))
                    yukfft[resp] = np.zeros(np.sum(ginds))
                    gforce[resp] = np.zeros(len(posvec_avg))
                    yukforce[resp] = np.zeros(len(posvec_avg))
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

            ### Add these to arrays so we don't have to recompute every time
            temp_gdat[ax1][ax2][0] = gfft
            temp_gdat[ax1][ax2][1][lambind] = yukfft
            plot_forces[ax1][ax2][0] = gforce
            plot_forces[ax1][ax2][1][lambind] = yukforce


            ### Compute appropriate limts for the parameter alpha in our eventual
            ### least-squared minimization to find the best-fit alpha
            newalpha = 2 * np.mean( np.abs(datfft_avg) ) / np.mean( np.abs(yukfft) ) * 1.0*10**(-1)
            testalphas = np.linspace(-1.0*newalpha, newalpha, 51)


            chi_sqs = np.zeros(len(testalphas))
            diagchi_sqs = np.zeros(len(testalphas))

            ### At each value of the 'testalpha', compute a reduced chi^2 statistic
            ### and compile these statistics into an array
            for alphaind, testalpha in enumerate(testalphas):

                ### Initialize the non-reduced chi^2 and Ndof variables
                chi_sq = 0
                diagchi_sq = 0
                Ndof = 0

                ### Loop over X, Y and Z resposnes
                for resp in [0,1,2]:
                    if (ignoreX and resp == 0) or \
                       (ignoreY and resp == 1) or \
                       (ignoreZ and resp == 2):
                        continue

                    ### Compute the difference between the data and the expected modified
                    ### gravity at this position and with this value of lambda
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

                    ### Add the difference of the real and imaginary components of the 
                    ### bead's response, normalized by the error
                    chi_sq += ( np.sum( np.abs(re_diff)**2 / (0.5*(daterr_avg[resp]**2)) ) + \
                              np.sum( np.abs(im_diff)**2 / (0.5*(daterr_avg[resp]**2)) ) )
                    if diag:
                        diagchi_sq += ( np.sum( np.abs(diag_re_diff)**2 / \
                                                (0.5*(diagdaterr_avg[resp]**2)) ) + \
                                        np.sum( np.abs(diag_im_diff)**2 / \
                                                (0.5*(diagdaterr_avg[resp]**2)) ) )

                    Ndof += len(re_diff) + len(im_diff)

                ### Reduce chi^2 by (Ndof - 1)
                chi_sqs[alphaind] = chi_sq / (Ndof - 1)
                if diag:
                    diagchi_sqs[alphaind] = diagchi_sq / (Ndof - 1)

            ### Compute the max value of the statistic found in order to make a reasonable
            ### first guess as to the parameters of the parabola
            max_chi = np.max(chi_sqs)
            if diag:
                max_diagchi = np.max(diagchi_sqs)
            max_alpha = np.max(testalphas)

            p0 = [max_chi/max_alpha**2, 0, 1]
            if diag:
                diag_p0 = [max_diagchi/max_alpha**2, 0, 1]

            try:
                ### Plot the reduced chi^2 vs alpha if the debug flag plot_best_alpha is True
                if yuklambda == lambdas[0] or yuklambda == lambdas[-1] and plot_best_alpha:
                    plt.plot(testalphas, chi_sqs)
                    plt.ylabel('Reduced $\chi^2$ Statistic')
                    plt.xlabel('$\\alpha$ Parameter')

                ### Try fitting reduced chi^2 vs. alpha to a parabola to extract the minimum
                popt, pcov = opti.curve_fit(parabola, testalphas, chi_sqs, \
                                            p0=p0, maxfev=100000)
                if diag:
                    diagpopt, diagpcov = opti.curve_fit(parabola, testalphas, diagchi_sqs, \
                                                        p0=diag_p0, maxfev=1000000)
            except:
                print "Couldn't fit"
                popt = [0,0,0]
                popt[2] = np.mean(chi_sqs)

            ### Get all the important information from the fit
            best_fit = -0.5 * popt[1] / popt[0]
            min_chi = parabola(best_fit, *popt)
            chi95 = min_chi + con_val
            soln = solve_parabola(chi95, popt)
            alpha95 = soln[np.argmax(np.abs(soln))] # select the larger (worse) solution
            fit_err = alpha95 - best_fit

            #fit_err = best_fit * np.sqrt( (pcov[1,1] / popt[1]**2) + (pcov[0,0] / popt[0]**2) )

            best_fit_alphas[lambind] = best_fit
            best_fit_errs[lambind] = fit_err
            if diag:
                diag_best_fit_alphas[lambind] = -0.5 * diagpopt[1] / diagpopt[0]

            stop = time.time()
            #print 'func eval time: ', stop-start

            if plot_best_alpha:

                fig_best, axarr_best = plt.subplots(3,1,sharex='all',sharey='all')
                lab_dict = {0: 'X', 1: 'Y', 2: 'Z'}
                for resp in [0,1,2]:
                    ylab = lab_dict[resp] + ' Force [N]'
                    axarr_best[resp].set_ylabel(ylab)
                    yforce = (yukforce[resp]-np.mean(yukforce[resp])) * best_fit_alphas[lambind]
                    axarr_best[resp].plot(posvec_avg, yforce, color='r', label='Best-Fit Force Curve')
                    axarr_best[resp].plot(binned_avg[resp][0], binned_avg[resp][1], color='k', \
                                          label='Averaged Response at One Position')
                axarr_best[2].set_xlabel('Y-position [um]')
                axarr_best[0].legend()
                plt.show()

        outdat[bias][ax1][ax2] = [best_fit_alphas, best_fit_errs, diag_best_fit_alphas]

    return outdat






def save_alphadat(outname, alphadat, lambdas, p0_bead):
    '''Saves all information relevant to the last step of the analysis. This
       way you can modify your statistical interpretation without reloading 
       possibly thousands of files.

       INPUTS:   outname,     output filename
                 alphadat,    output from find_alpha_vs_file()
                 lambdas,     array of lambda values given to find_alpha_vs_file()
                 minsep,      separation at maximum throw
                 maxthrow,    maximum throw of attractor (toward bead)
                 beadheight,  height of bead in cantilever coordinates

       OUTPUTS:  soln, tuple with solution (x_small, x_large).
    '''
    dump = {}
    dump['alphadat'] = alphadat
    dump['lambdas'] = lambdas
    dump['p0_bead'] = p0_bead

    pickle.dump(dump, open(outname, 'wb'))


def load_alphadat(filename):
    '''Loads the pickled alphadat and returns the dictionary. The keys make
       sense so there's really no reason to unpack more'''
    stuff = pickle.load(open(filename, 'rb'))
    return stuff






def fit_alpha_vs_alldim(alphadat, lambdas, p0_bead=[20,0,20], \
                        plot=False, weight_planar=True):
    '''Takes the best_fit_alphas vs height and separation, and fits them to a plane,
       for each value of lambda. Extracts some idea of sensitivity from this fit

       INPUTS:   alphadat,       output from find_alpha_vs_file()
                 lambdas,        array of lambda values given to find_alpha_vs_file()
                 minsep,         separation at maximum throw
                 maxthrow,       maximum throw of attractor (toward bead)
                 beadheight,     height of bead in cantilever coordinates
                 plot,           boolean to plot planar fitting for debug
                 weight_planar,  weight the planar fit by the confidence 
                                    level on the best fit alpha

       OUTPUTS:  fits,     
                 outdat, 
                 lambdas, 
                 alphas_1,      HAVEN'T FINALIZED THESE. COMMENTS TO COME
                 alphas_2, 
                 alphas_3, 
    '''

    biasvec = alphadat.keys()
    ax1vec = alphadat[biasvec[0]].keys()
    ax2vec = alphadat[biasvec[0]][ax1vec[0]].keys()

    ### Assume separations are encoded in ax1 and heights in ax2
    seps = 80 + p0_bead[0] - np.array(ax1vec)
    heights = p0_bead[2] - np.array(ax2vec) 

    ### Sort the heights and separations and build a grid
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

        ### Fit to a plane for each value of lambda
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

            ### Since the data dictionary was indexed by cantilever settings 
            ### rather than actual bead positions, we have to sort the data
            ### to match the sorted separations and heights
            dat = dat[sort1,:]
            dat = dat[:,sort2]

            errs = errs[sort1,:]
            errs = errs[:,sort2]

            scale_fac = np.mean(dat)

            dat_sc = dat * (1.0 / scale_fac)
            errs_sc = errs * (1.0 / scale_fac)
            if not weight_planar:
                errs_sc = np.ones_like(dat_sc)

            ### Defined a funciton to minize via least-squared optimization
            def func(params, fdat=dat_sc, ferrs=errs_sc):
                funcval = params[0] * heights_g + params[1] * seps_g + params[2]
                return ((funcval - fdat) / ferrs).flatten()

            ### Optimize the previously defined function
            res = opti.leastsq(func, [0.2*np.mean(dat_sc), 0.2*np.mean(dat_sc), 0], \
                               full_output=1, maxfev=10000)

            try:
                x = res[0]
                residue = linalg.inv(res[1])[2,2]
            except:
                2+2


            ### Deplane the data and extract some statistics
            deplaned = dat - scale_fac * (x[0] * heights_g + x[1] * seps_g + x[2])
            deplaned_wconst = deplaned + scale_fac * x[2]

            deplaned_avg = np.mean(deplaned)
            deplaned_std = np.std(deplaned) / dat.size

            ### DEBUG CLAUSE: checks to make sure alphadat isn't changing
            ### crazily between different values of lambda
            #if lambind == 0:
            #    old_dat = dat
            #    diff = 0.0
            #else:
            #    diff = (dat - old_dat) / dat
            #    old_dat = dat
            #print lambind, np.mean(dat), np.mean(diff), x[2]

            if plot:
                
                major_ticks = np.arange(15, 36, 5)

                vals = x[0] * heights_g + x[1] * seps_g + x[2]
                vals = vals * scale_fac

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(heights_g, seps_g, vals, rstride=1, cstride=1, alpha=0.3, \
                                color='r')
                ax.scatter(heights_g, seps_g, dat, label='Best-fit alphas')
                ax.scatter(heights_g, seps_g, errs, label='Errors for weighting in planar fit')
                ax.legend()
                ax.set_xlabel('Z-position [um]')
                ax.set_ylabel('X-separation [um]')
                ax.set_yticks(major_ticks)
                ax.set_zlabel('Alpha [arb]')

                fig2 = plt.figure()
                ax2 = fig2.gca(projection='3d')
                ax2.scatter(heights_g, seps_g, deplaned_wconst, label='De-planed')
                ax2.legend()
                ax2.set_xlabel('Z-position [um]')
                ax2.set_ylabel('X-separation [um]')
                ax2.set_yticks(major_ticks)
                ax2.set_zlabel('Alpha [arb]')
                print np.mean(deplaned_wconst), np.log10(np.mean(deplaned_wconst))

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



