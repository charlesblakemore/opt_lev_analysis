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

only_closest = False #True
minsep = 15       # um
maxthrow = 80     # um
beadheight = 10   # um

#data_dir = '/data/20180314/bead1/grav_data/ydrive_1sep_1height_extdrive_nofield_long'
#data_dir = '/data/20180314/bead1/grav_data/ydrive_1sep_1height_nofield_shieldin'
#data_dir = '/data/20180314/bead1/grav_data/ydrive_1sep_1height_1V-1300Hz_shieldin_0mV-cant'
#data_dir = '/data/20180314/bead1/grav_data/ydrive_1sep_1height_2V-2200Hz_shield_0mV-cant'

data_dir = '/data/20180314/bead1/grav_data/ydrive_6sep_1height_shield-2Vac-2200Hz_cant-0mV'

#savepath = '/sensitivities/20180314_grav_shield-2200Hz_cant-m100mV_allharm.npy'
savepath = '/sensitivities/20180314_grav_shieldin-2V-2200Hz_cant-0V_allharm.npy'
save = False
load = False
file_inds = (0, 10)

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

tfdate = '' #'20180215'
diag = False

confidence_level = 0.95

lamb_range = (1.7e-6, 1e-4)

#user_lims = [(65e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
user_lims = [(5e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
#user_lims = []

tophatf = 300   # Hz, doesn't reconstruct data above this frequency
nharmonics = 10
harms = [1,3,5,7]

plot_just_current = False
figtitle = ''

ignoreX = False
ignoreY = False
ignoreZ = False

compute_min_alpha = False

##################################################################
################# Constraints to plot against ####################

alpha_plot_lims = (1000, 10**13)
lambda_plot_lims = (10**(-7), 10**(-4))


#limitdata_path = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
#                 'decca2_limit.txt'

limitdata_path = '/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'


#limitdata_path2 = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
#                 'no_decca2_limit.txt'
limitdata_path2 = '/sensitivities/decca2_limits.txt'
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








def get_data_at_harms(files, gfuncs, yukfuncs, lambdas, lims, \
                      minsep=20, maxthrow=80, beadheight=5,\
                      cantind=0, ax1='x', ax2='z', diag=True, plottf=False, \
                      width=0, nharmonics=10, harms=[], \
                      ext_cant_drive=False, ext_cant_ind=1, \
                      ignoreX=False, ignoreY=False, ignoreZ=False,  noiseband=10):
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
        
        if cantbias not in list(fildat.keys()):
            fildat[cantbias] = {}
        if ax1pos not in list(fildat[cantbias].keys()):
            fildat[cantbias][ax1pos] = {}
        if ax2pos not in list(fildat[cantbias][ax1pos].keys()):
            fildat[cantbias][ax1pos][ax2pos] = []

        if ax1pos not in list(temp_gdat.keys()):
            temp_gdat[ax1pos] = {}
        if ax2pos not in list(temp_gdat[ax1pos].keys()):
            temp_gdat[ax1pos][ax2pos] = [[], []]
            temp_gdat[ax1pos][ax2pos][1] = [[]] * len(lambdas)

        cfind = len(fildat[cantbias][ax1pos][ax2pos])
        fildat[cantbias][ax1pos][ax2pos].append([])

        if fil_ind == 0 and plottf:
            df.diagonalize(date=tfdate, maxfreq=tophatf, plot=True)
        else:
            df.diagonalize(date=tfdate, maxfreq=tophatf)

        if fil_ind == 0:
            ginds, fund_ind, drive_freq, drive_ind = \
                df.get_boolean_cantfilt(ext_cant_drive=ext_cant_drive, ext_cant_ind=ext_cant_ind, \
                                        nharmonics=nharmonics, harms=harms, width=width)

        datffts, diagdatffts, daterrs, diagdaterrs = \
                    df.get_datffts_and_errs(ginds, drive_freq, noiseband=noiseband, plot=False, \
                                            diag=diag)

        drivevec = df.cant_data[drive_ind]
        
        mindrive = np.min(drivevec)
        maxdrive = np.max(drivevec)

        posvec = np.linspace(mindrive, maxdrive, 500)
        ones = np.ones_like(posvec)

        start = time.time()
        for lambind, yuklambda in enumerate(lambdas):

            if ax1 == 'x' and ax2 == 'z':
                newxpos = minsep + (maxthrow - ax1pos)
                newheight = ax2pos - beadheight
            elif ax1 =='z' and ax2 == 'x':
                newxpos = minsep + (maxthrow - ax2pos)
                newheight = ax1pos - beadheight
            else:
                print("Coordinate axes don't make sense for gravity data...")
                print("Proceeding anyway, but results might be hard to interpret")
                newxpos = ax1pos
                newheight = ax2pos

            if (newxpos < lims[0][0]*1e6) or (newxpos > lims[0][1]*1e6):
                #print 'skipped x'
                continue

            if (newheight < lims[2][0]*1e6) or (newheight > lims[2][1]*1e6):
                #print 'skipped z'
                continue

            pts = np.stack((newxpos*ones, posvec, newheight*ones), axis=-1)

            gfft = [[], [], []]
            yukfft = [[], [], []]
            for resp in [0,1,2]:
                if (ignoreX and resp == 0) or (ignoreY and resp == 1) or (ignoreZ and resp == 2):
                    gfft[resp] = np.zeros(np.sum(ginds))
                    yukfft[resp] = np.zeros(np.sum(ginds))
                    continue

                if len(temp_gdat[ax1pos][ax2pos][0]):
                    gfft[resp] = temp_gdat[ax1pos][ax2pos][0][resp]
                else:
                    gforcevec = gfuncs[resp](pts*1e-6)
                    gforcefunc = interp.interp1d(posvec, gforcevec)
                    gforcet = gforcefunc(drivevec)

                    gfft[resp] =  np.fft.rfft(gforcet)[ginds]

                if len(temp_gdat[ax1pos][ax2pos][1][lambind]):
                    yukfft[resp] = temp_gdat[ax1pos][ax2pos][1][lambind][resp]
                else:
                    yukforcevec = yukfuncs[resp][lambind](pts*1e-6)
                    yukforcefunc = interp.interp1d(posvec, yukforcevec)
                    yukforcet = yukforcefunc(drivevec)

                    yukfft[resp] = np.fft.rfft(yukforcet)[ginds]

            gfft = np.array(gfft)
            yukfft = np.array(yukfft)

            temp_gdat[ax1pos][ax2pos][0] = gfft
            temp_gdat[ax1pos][ax2pos][1][lambind] = yukfft

            outdat = (yuklambda, datffts, diagdatffts, daterrs, diagdaterrs, gfft, yukfft)

            fildat[cantbias][ax1pos][ax2pos][cfind].append(outdat)

        stop = time.time()
        #print 'func eval time: ', stop-start

    return fildat













def get_alpha_lambda(fildat, diag=True, ignoreX=False, ignoreY=False, ignoreZ=False, \
                     plot=True, save=False, savepath='', confidence_level=0.95, \
                     only_closest=False, ax1='x', ax2='z', lamb_range=(1e-9, 1e-2)):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then performs an optimal filter using the cantilever drive and 
       a theoretical force vs position to generate the filter/template.
       The result of the optimal filtering is stored, and the data 
       released from memory

       INPUTS: fildat

       OUTPUTS: 
    '''

    # For the confidence interval, compute the inverse CDF of a 
    # chi^2 distribution at given confidence level and compare to 
    # liklihood ratio via a goodness of fit parameter.
    # Refer to scipy.stats documentation to understand chi2
    chi2dist = stats.chi2(1)
    # factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
    con_val = 0.5 * chi2dist.ppf(confidence_level)

    colors = bu.get_colormap(len(lambdas))

    alphas = np.zeros_like(lambdas)
    diagalphas = np.zeros_like(lambdas)
    testalphas = np.linspace(-10**10, 10**10, 11)

    minalphas = [[]] * len(lambdas)

    biasvec = list(fildat.keys())
    biasvec.sort()
    ax1posvec = list(fildat[biasvec[0]].keys())
    ax1posvec.sort()
    ax2posvec = list(fildat[biasvec[0]][ax1posvec[0]].keys())
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
        

    newlamb = lambdas[(lambdas > lamb_range[0]) * (lambdas < lamb_range[-1])]
    tot_iterations = len(biasvec) * len(ax1posvec) * len(ax2posvec) * \
                         len(newlamb) * len(testalphas) + 1
    i = -1

    # To test chi2 fit against "fake" data, uncomment these lines
    rands = np.random.randn(*fildat[biasvec[0]][ax1posvec[0]][ax2posvec[0]][0][0][1].shape)
    rands2 = np.random.randn(*fildat[biasvec[0]][ax1posvec[0]][ax2posvec[0]][0][0][1].shape)

    for lambind, yuklambda in enumerate(lambdas):
        #if lambind != 48:
        #    continue

        if (yuklambda < lamb_range[0]) or (yuklambda > lamb_range[1]):
            continue

        test = fildat[biasvec[0]][ax1posvec[0]][ax2posvec[0]][0][lambind]
        test_yukdat = test[-1]
        test_dat = test[1]

        newalpha = 1e-4 * np.sqrt(np.mean(np.abs(test_dat) / np.abs(test_yukdat)))
        testalphas = np.linspace(-1.0*newalpha, newalpha, 21)

        chi_sqs = np.zeros(len(testalphas))
        diagchi_sqs = np.zeros(len(testalphas))

        for alphaind, testalpha in enumerate(testalphas):
            N = 0
            chi_sq = 0
            diagchi_sq = 0

            for bias, ax1pos, ax2pos in itertools.product(biasvec, ax1posvec, ax2posvec):
                i += 1
                bu.progress_bar(i, tot_iterations, suffix=' Fitting the Data for Chi^2')

                for fil_ind in range(len(fildat[bias][ax1pos][ax2pos])):
                    dat = fildat[bias][ax1pos][ax2pos][fil_ind][lambind]
                    assert dat[0] == yuklambda
                    _, datfft, diagdatfft, daterr, diagdaterr, gfft, yukfft = dat

                    # To test chi2 fit against "fake" data, uncomment these lines
                    #datfft = yukfft * -0.5e9
                    #datfft += (1.0 / np.sqrt(2)) * daterr * rands + \
                    #          (1.0 / np.sqrt(2)) * daterr * rands2 * 1.0j
                    #gfft = np.zeros_like(datfft)

                    for resp in [0,1,2]:
                        if (ignoreX and resp == 0) or \
                           (ignoreY and resp == 1) or \
                           (ignoreZ and resp == 2):
                            print(ignoreX, ignoreY, ignoreZ, resp)
                            continue
                        re_diff = datfft[resp].real - \
                                  (gfft[resp].real + testalpha * yukfft[resp].real )
                        im_diff = datfft[resp].imag - \
                                  (gfft[resp].imag + testalpha * yukfft[resp].imag )
                        if diag:
                            diag_re_diff = diagdatfft[resp].real - \
                                           (gfft[resp].real + testalpha * yukfft[resp].real )
                            diag_im_diff = diagdatfft[resp].imag - \
                                           (gfft[resp].imag + testalpha * yukfft[resp].imag )

                        #plt.plot(np.abs(re_diff))
                        #plt.plot(daterr[resp])
                        #plt.show()

                        chi_sq += ( np.sum( np.abs(re_diff)**2 / (0.5*daterr[resp]**2) ) + \
                                  np.sum( np.abs(im_diff)**2 / (0.5*daterr[resp]**2) ) )
                        if diag:
                            diagchi_sq += ( np.sum( np.abs(diag_re_diff)**2 / \
                                                    (0.5*diagdaterr[resp]**2) ) + \
                                            np.sum( np.abs(diag_im_diff)**2 / \
                                                    (0.5*diagdaterr[resp]**2) ) )

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

        #if lambind == 0:
        #    p0 = [0.15e9, 0, 5]
        #else:
        #    p0 = p0_old

        if plot:
            plt.figure(1)
            plt.plot(testalphas, chi_sqs, color = colors[lambind])
            if diag:
                plt.figure(2)
                plt.plot(testalphas, diagchi_sqs, color = colors[lambind])
    
        try:
            popt, pcov = opti.curve_fit(parabola, testalphas, chi_sqs, \
                                            p0=p0, maxfev=100000)
            if diag:
                diagpopt, diagpcov = opti.curve_fit(parabola, testalphas, diagchi_sqs, \
                                                    p0=diag_p0, maxfev=1000000)
        except:
            print("Couldn't fit")
            popt = [0,0,0]
            popt[2] = np.mean(chi_sqs)

        regular_con_val = con_val + np.min(chi_sqs)
        if diag:
            diag_con_val = con_val + np.min(diagchi_sqs)

        # Select the positive root for the non-diagonalized data
        soln1 = ( -1.0 * popt[1] + np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - regular_con_val)) ) / (2 * popt[0])
        soln2 = ( -1.0 * popt[1] - np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - regular_con_val)) ) / (2 * popt[0])

        if diag:
            diagsoln1 = ( -1.0 * diagpopt[1] + np.sqrt( diagpopt[1]**2 - \
                            4 * diagpopt[0] * (diagpopt[2] - diag_con_val)) ) / (2 * diagpopt[0])
            diagsoln2 = ( -1.0 * diagpopt[1] - np.sqrt( diagpopt[1]**2 - \
                            4 * diagpopt[0] * (diagpopt[2] - diag_con_val)) ) / (2 * diagpopt[0])

        if soln1 > soln2:
            alpha_con = soln1
        else:
            alpha_con = soln2

        if diag:
            if diagsoln1 > diagsoln2:
                diagalpha_con = diagsoln1
            else:
                diagalpha_con = diagsoln2

        alphas[lambind] = alpha_con
        if diag:
            diagalphas[lambind] = alpha_con


    if plot:
        plt.figure(1)
        plt.title('Goodness of Fit for Various Lambda', fontsize=16)
        plt.xlabel('Alpha Parameter [arb]', fontsize=14)
        plt.ylabel('$\chi^2$', fontsize=18)

        if diag:
            plt.figure(2)
            plt.title('Goodness of Fit for Various Lambda - DIAG', fontsize=16)
            plt.xlabel('Alpha Parameter [arb]', fontsize=14)
            plt.ylabel('$\chi^2$', fontsize=18)

        plt.show()

    if not diag:
        diagalphas = np.zeros_like(alphas)

    if save:
        if savepath == '':
            print('No save path given, type full path here')
            savepath = input('path: ')
        
        np.save(savepath, [lambdas, alphas, diagalphas])


    return lambdas, alphas, diagalphas





















def get_alpha_vs_file(fildat, diag=True, ignoreX=False, ignoreY=False, ignoreZ=False, \
                     plot=True, save=False, savepath='', confidence_level=0.95, \
                     only_closest=False, ax1='x', ax2='z', lamb_range=(1e-9, 1e-2)):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then performs an optimal filter using the cantilever drive and 
       a theoretical force vs position to generate the filter/template.
       The result of the optimal filtering is stored, and the data 
       released from memory

       INPUTS: fildat

       OUTPUTS: 
    '''

    # For the confidence interval, compute the inverse CDF of a 
    # chi^2 distribution at given confidence level and compare to 
    # liklihood ratio via a goodness of fit parameter.
    # Refer to scipy.stats documentation to understand chi2
    chi2dist = stats.chi2(1)
    # factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
    con_val = 0.5 * chi2dist.ppf(confidence_level)

    colors = bu.get_colormap(len(lambdas))

    alphas = np.zeros_like(lambdas)
    diagalphas = np.zeros_like(lambdas)
    testalphas = np.linspace(-10**10, 10**10, 11)

    biasvec = list(fildat.keys())
    biasvec.sort()
    ax1posvec = list(fildat[biasvec[0]].keys())
    ax1posvec.sort()
    ax2posvec = list(fildat[biasvec[0]][ax1posvec[0]].keys())
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
        

    newlamb = lambdas[(lambdas > lamb_range[0]) * (lambdas < lamb_range[-1])]
    tot_iterations = len(biasvec) * len(ax1posvec) * len(ax2posvec) * len(newlamb) * len(testalphas)
    i = -1

    for lambind, yuklambda in enumerate(lambdas):
        if lambind != 48:
            continue

        if (yuklambda < lamb_range[0]) or (yuklambda > lamb_range[1]):
            continue

        test = fildat[biasvec[0]][ax1posvec[0]][ax2posvec[0]][0][lambind]
        test_yukdat = test[-1]
        test_dat = test[1]
        
        newalpha = 1e-4 * np.sqrt(np.mean(np.abs(test_dat) / np.abs(test_yukdat)))
        testalphas = np.linspace(-1.0*newalpha, newalpha, 11)

        for bias, ax1pos, ax2pos in itertools.product(biasvec, ax1posvec, ax2posvec):
            i += 1
            bu.progress_bar(i, tot_iterations)
 
            minalphas = [0] * len(fildat[bias][ax1pos][ax2pos])
            diag_minalphas = [0] * len(fildat[bias][ax1pos][ax2pos])

            for fil_ind in range(len(fildat[bias][ax1pos][ax2pos])):
                dat = fildat[bias][ax1pos][ax2pos][fil_ind][lambind]
                assert dat[0] == yuklambda
                _, datfft, diagdatfft, daterr, diagdaterr, gfft, yukfft = dat

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
                        re_diff = datfft[resp].real - \
                                  (gfft[resp].real + testalpha * yukfft[resp].real )
                        im_diff = datfft[resp].imag - \
                                  (gfft[resp].imag + testalpha * yukfft[resp].imag )
                        if diag:
                            diag_re_diff = diagdatfft[resp].real - \
                                           (gfft[resp].real + testalpha * yukfft[resp].real )
                            diag_im_diff = diagdatfft[resp].imag - \
                                           (gfft[resp].imag + testalpha * yukfft[resp].imag )

                        #plt.plot(np.abs(re_diff))
                        #plt.plot(daterr[resp])
                        #plt.show()

                        chi_sq += ( np.sum( np.abs(re_diff)**2 / (0.5*(daterr[resp]**2)) ) + \
                                  np.sum( np.abs(im_diff)**2 / (0.5*(daterr[resp]**2)) ) )
                        if diag:
                            diagchi_sq += ( np.sum( np.abs(diag_re_diff)**2 / \
                                                    (0.5*(diagdaterr[resp]**2)) ) + \
                                            np.sum( np.abs(diag_im_diff)**2 / \
                                                    (0.5*(diagdaterr[resp]**2)) ) )

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
                    popt, pcov = opti.curve_fit(parabola, testalphas, chi_sqs, \
                                                p0=p0, maxfev=100000)
                    if diag:
                        diagpopt, diagpcov = opti.curve_fit(parabola, testalphas, diagchi_sqs, \
                                                            p0=diag_p0, maxfev=1000000)
                except:
                    print("Couldn't fit")
                    popt = [0,0,0]
                    popt[2] = np.mean(chi_sqs)

                regular_con_val = con_val + np.min(chi_sqs)
                if diag:
                    diag_con_val = con_val + np.min(diagchi_sqs)

                # Select the positive root for the non-diagonalized data
                soln1 = ( -1.0 * popt[1] + np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - regular_con_val)) ) / (2 * popt[0])
                soln2 = ( -1.0 * popt[1] - np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - regular_con_val)) ) / (2 * popt[0])

                if diag:
                    diagsoln1 = ( -1.0 * diagpopt[1] + np.sqrt( diagpopt[1]**2 - \
                            4 * diagpopt[0] * (diagpopt[2] - diag_con_val)) ) / (2 * diagpopt[0])
                    diagsoln2 = ( -1.0 * diagpopt[1] - np.sqrt( diagpopt[1]**2 - \
                            4 * diagpopt[0] * (diagpopt[2] - diag_con_val)) ) / (2 * diagpopt[0])

                if soln1 > soln2:
                    alpha_con = soln1
                else:
                    alpha_con = soln2

                if diag:
                    if diagsoln1 > diagsoln2:
                        diagalpha_con = diagsoln1
                    else:
                        diagalpha_con = diagsoln2

                minalphas[fil_ind] = alpha_con
                if diag:
                    diag_minalphas[fil_ind] = diagalpha_con


            if plot:
                minfig, minaxarr = plt.subplots(1,2,figsize=(10,5),dpi=150)
                minaxarr[0].plot(minalphas)
                minaxarr[0].set_title('Min $\\alpha$ vs. Time', fontsize=18)
                minaxarr[0].set_xlabel('File Num', fontsize=16)
                minaxarr[0].set_ylabel('$\\alpha$ [arb]', fontsize=16)

                minaxarr[1].hist(minalphas, bins=20)
                minaxarr[1].set_xlabel('$\\alpha$ [arb]', fontsize=16)

                plt.tight_layout()
                plt.show()


    return minalphas



















if not plot_just_current:
    gfuncs, yukfuncs, lambdas, lims = build_mod_grav_funcs(theory_data_dir)

    datafiles = bu.find_all_fnames(data_dir, ext=config.extensions['data'])

    datafiles = datafiles[file_inds[0]:file_inds[1]]

    if len(datafiles) == 0:
        print("Found no files in: ", data_dir)
        quit()


    fildat = get_data_at_harms(datafiles, gfuncs, yukfuncs, lambdas, lims, \
                               minsep=minsep, maxthrow=maxthrow, beadheight=beadheight, \
                               cantind=0, ax1='x', ax2='z', diag=diag, plottf=False, \
                               nharmonics=nharmonics, harms=harms, \
                               ext_cant_drive=True, ext_cant_ind=1, \
                               ignoreX=ignoreX, ignoreY=ignoreY, ignoreZ=ignoreZ)
    
    if compute_min_alpha:
        _ = get_alpha_vs_file(fildat, only_closest=only_closest, \
                              ignoreX=ignoreX, ignoreY=ignoreY, ignoreZ=ignoreZ, \
                              lamb_range=lamb_range, diag=diag, plot=True)

    newlambdas, alphas, diagalphas = \
                    get_alpha_lambda(fildat, only_closest=only_closest, \
                                     ignoreX=ignoreX, ignoreY=ignoreY, ignoreZ=ignoreZ, \
                                     lamb_range=lamb_range, diag=diag)


    outdat = [newlambdas, alphas, diagalphas]
    if save:
        np.save(savepath, outdat)

    if load:
        dat = np.load(savepath)
        newlambdas = dat[0]
        alphas = dat[1]
        diagalphas = dat[2]










fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)
if diag:
    fig2, ax2 = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

if not plot_just_current:
    ax.loglog(newlambdas, alphas, linewidth=2, label='95% CL')
    if diag:
        ax2.loglog(newlambdas, diagalphas, linewidth=2, label='95% CL')

ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
ax.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('$\\alpha$')

ax.legend(numpoints=1, fontsize=9)

ax.set_title(figtitle)

plt.tight_layout()

if diag:
    ax2.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
    ax2.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
    ax2.grid()

    ax2.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
    ax2.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

    ax2.set_xlabel('$\lambda$ [m]')
    ax2.set_ylabel('$\\alpha$')

    ax2.legend(numpoints=1, fontsize=9)

    ax2.set_title(figtitle)

    plt.tight_layout()

plt.show()
