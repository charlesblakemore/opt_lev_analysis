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


##################################################################
######################## Script Params ###########################

minsep = 15       # um
maxthrow = 80     # um
beadheight = 10   # um

#data_dir = '/data/20180314/bead1/grav_data/ydrive_6sep_1height_shield-2Vac-2200Hz_cant-0mV'
#data_dir = '/data/20180524/bead1/grav_data/many_sep_many_h'

#data_dir = '/data/20180613/bead1/grav_data/no_shield/X60-80um_Z20-30um'
data_dir = '/data/20180613/bead1/grav_data/shield/X70-80um_Z15-25um'

savepath = '/sensitivities/20180613_grav-no-shield_1.npy'
save = False
load = False
file_inds = (0, 2000)
max_file_per_pos = 1

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'
#theory_data_dir = '/data/grav_sim_data/1um_spacing_x-0-p80_y-m250-p250_z-m20-p20/'

tfdate = ''
diag = False

confidence_level = 0.95

lamb_range = (1.7e-6, 1e-4)

#user_lims = [(65e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
user_lims = [(5e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
#user_lims = []

tophatf = 300   # Hz, doesn't reconstruct data above this frequency
nharmonics = 10
harms = [1,2,3,4,5,6,7,8,9]

plotfilt = False
plot_just_current = False
figtitle = ''

ignoreX = False
ignoreY = False
ignoreZ = False

compute_min_alpha = False

noiseband = 10

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








def get_data_at_harms(files, minsep=20, maxthrow=80, beadheight=5,\
                      cantind=0, ax1='x', ax2='z', diag=True, plottf=False, \
                      width=0, nharmonics=10, harms=[], \
                      ext_cant_drive=False, ext_cant_ind=1, plotfilt=False, \
                      max_file_per_pos=1000):
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

        if (newxpos < lims[0][0]*1e6) or (newxpos > lims[0][1]*1e6):
            #print 'skipped x'
            continue

        if (newheight < lims[2][0]*1e6) or (newheight > lims[2][1]*1e6):
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

        drivevec = df.cant_data[drive_ind]
        
        mindrive = np.min(drivevec)
        maxdrive = np.max(drivevec)

        posvec = np.linspace(mindrive, maxdrive, 500)
        ones = np.ones_like(posvec)

        pts = np.stack((newxpos*ones, posvec, newheight*ones), axis=-1)

        fildat[cantbias][ax1pos][ax2pos].append((drivevec, posvec, pts, ginds, \
                                                 datffts, diagdatffts, daterrs, diagdaterrs))

    return fildat









def find_alpha_vs_file(fildat, gfuncs, yukfuncs, lambdas, lims, \
                      ignoreX=False, ignoreY=False, ignoreZ=False):
    '''Loops over the output from get_data_at_harms, fits each set of
       data against the appropriate modified gravity template and compiles
       the result into a dictionary

       INPUTS: 

       OUTPUTS: 
    '''

    outdat = {}
    temp_gdat = {}

    biasvec = fildat.keys()
    ax1vec = fildat[biasvec[0]].keys()
    ax2vec = fildat[biasvec[0]][ax1vec[0]].keys()

    for bias in biasvec:
        outdat[bias] = {}
        for ax1 in ax1vec:
            outdat[bias][ax1] = {}
            temp_gdat[ax1] = {}
            for ax2 in ax2vec:
                outdat[bias][ax1][ax2] = []
                temp_gdat[ax1][ax2] = [[],[]]
                temp_gdat[ax1][ax2][1] = [[]] * len(lambdas)


    i = 0
    totlen = len(biasvec) * len(ax1vec) * len(ax2vec)
    for bias, ax1, ax2 in itertools.product(biasvec, ax1vec, ax2vec):
        i += 1
        suff = '%i / %i param combinations' % (i, totlen)
        if i != totlen:
            newline=False
        else:
            newline=True

        dat = fildat[bias][ax1][ax2]
        outdat[bias][ax1][ax2] = []
        
        j = 0
        for fil in dat:
            drivevec, posvec, pts, ginds, datfft, diagdatfft, daterr, diagdaterr = fil

            start = time.time()

            ######################################
            ######################################

            best_fit_alphas = np.zeros(len(lambdas))
            diag_best_fit_alphas = np.zeros(len(lambdas))

            for lambind, yuklambda in enumerate(lambdas):
                j += 1
                bu.progress_bar(j, len(lambdas) * len(dat), suffix=suff, newline=newline)

                gfft = [[], [], []]
                yukfft = [[], [], []]
                for resp in [0,1,2]:
                    if (ignoreX and resp == 0) or (ignoreY and resp == 1) or (ignoreZ and resp == 2):
                        gfft[resp] = np.zeros(np.sum(ginds))
                        yukfft[resp] = np.zeros(np.sum(ginds))
                        continue

                    if len(temp_gdat[ax1][ax2][0]):
                        gfft[resp] = temp_gdat[ax1][ax2][0][resp]
                    else:
                        gforcevec = gfuncs[resp](pts*1e-6)
                        gforcefunc = interp.interp1d(posvec, gforcevec)
                        gforcet = gforcefunc(drivevec)

                        gfft[resp] =  np.fft.rfft(gforcet)[ginds]

                    if len(temp_gdat[ax1][ax2][1][lambind]):
                        yukfft[resp] = temp_gdat[ax1][ax2][1][lambind][resp]
                    else:
                        yukforcevec = yukfuncs[resp][lambind](pts*1e-6)
                        yukforcefunc = interp.interp1d(posvec, yukforcevec)
                        yukforcet = yukforcefunc(drivevec)

                        yukfft[resp] = np.fft.rfft(yukforcet)[ginds]

                gfft = np.array(gfft)
                yukfft = np.array(yukfft)

                temp_gdat[ax1][ax2][0] = gfft
                temp_gdat[ax1][ax2][1][lambind] = yukfft


                newalpha = 2 * np.mean( np.abs(datfft) ) / np.mean( np.abs(yukfft) ) * 10**(-4)
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
                    print "Couldn't fit"
                    popt = [0,0,0]
                    popt[2] = np.mean(chi_sqs)

                best_fit_alphas[lambind] = -2.0 * popt[1] / popt[0]
                if diag:
                    diag_best_fit_alphas[lambind] = -2.0 * diagpopt[1] / diagpopt[0]

                stop = time.time()
                #print 'func eval time: ', stop-start

            outdat[bias][ax1][ax2].append([best_fit_alphas, diag_best_fit_alphas])

    return outdat






def fit_alpha_vs_sep_1height(alphadat, height, minsep=10.0, maxthrow=80.0, beadheight=40.0):
    fits = {}

    biasvec = alphadat.keys()
    ax1vec = alphadat[biasvec[0]].keys()
    ax2vec = alphadat[biasvec[0]][ax1vec[0]].keys()

    ### Assume separations are encoded in ax1 and heights in ax2
    seps = maxthrow + minsep - np.array(ax1vec)
    heights = np.array(ax2vec)- beadheight

    ax2ind = np.argmin(np.abs(heights - height))

    testvec = [] 
    ### Perform identical fits for each cantilever bias and height
    for bias in biasvec:
        fits[bias] = {}
        for ax2pos in ax2vec:
            if ax2pos != ax2vec[ax2ind]:
                continue
            fits[bias][ax2pos] = []

            ### Fit alpha vs separation for each value of yuklambda
            for lambind, yuklambda in enumerate(lambdas):
                dat = []

                ### Loop over all files at each separation and collect
                ### the value of alpha for the current value of yuklambda
                for ax1ind, ax1pos in enumerate(ax1vec):
                    for fil in alphadat[bias][ax1pos][ax2pos]:
                        dat.append([seps[ax1ind], fil[0][lambind]])

                ### Sort data for a monotonically increasing separation
                dat = np.array(dat)
                sort_inds = np.argsort(dat[:,0])
                dat = dat[sort_inds]

                SCALE_FAC = 1.0*10**9

                ### Fit alpha vs. separation (most naive fit with a line)
                popt, pcov = opti.curve_fit(line, dat[:,0], dat[:,1] * (1.0 / SCALE_FAC), \
                                            maxfev=10000)

                testvec.append([popt[1]*SCALE_FAC, np.sqrt(pcov[1,1])*SCALE_FAC])

                print popt * SCALE_FAC

                plt.plot(dat[:,0], dat[:,1], '.', ms=5, label='Best Fit Alphas')
                plt.plot(dat[:,0], line(dat[:,0], popt[0]*SCALE_FAC, popt[1]*SCALE_FAC), \
                         label='Linear + Constant Fit')
                plt.xlabel('Separation [um]')
                plt.ylabel('Alpha [abs]')
                plt.legend()
                plt.tight_layout()
                plt.show()

                ### Append the result to our output array
                fits[bias][ax2pos].append(popt*SCALE_FAC)


    testvec = np.array(testvec)
    #plt.loglog(lambdas, np.abs(testvec[:,0]), label='Best fit alpha')
    #plt.loglog(lambdas, 2*np.abs(testvec[:,1]), label='95% CL (assuming IID error)')
    #plt.legend()
    #plt.show()

    alphas_bf = np.abs(testvec[:,0])
    alphas_95cl = 2.0 * np.abs(testvec[:,1])

    return alphas_bf, alphas_95cl, fits







def fit_alpha_vs_alldim(alphadat, lambdas, minsep=10.0, maxthrow=80.0, beadheight=40.0, \
                        plot=False):

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
                    tempdat = []

                    for fil in alphadat[bias][ax1pos][ax2pos]:
                        tempdat.append(fil[0][lambind])

                    dat[ax1ind][ax2ind] = np.mean(tempdat)
                    errs[ax1ind][ax2ind] = np.std(tempdat)

            dat = np.array(dat)
            errs = np.array(errs)

            dat = dat[sort1,:]
            dat = dat[:,sort2]

            errs = errs[sort1,:]
            errs = errs[:,sort2]

            def err_func(params):
                funcval = params[0] * seps_g + params[1] * heights_g + params[2]
                lst_sq = np.abs(dat - funcval)**2 / errs**2
                return np.sum(lst_sq)
                
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(seps_g, heights_g, dat)

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111, projection='3d')
                ax2.scatter(seps_g, heights_g, errs)

                plt.show()

            res = opti.minimize(err_func, [np.mean(dat), np.mean(dat), np.mean(dat)])

            fits[bias][lambind] = res
            outdat[bias][lambind] = (dat, errs)

    return fits, outdat











height = 25.0


if not plot_just_current:
    gfuncs, yukfuncs, lambdas, lims = build_mod_grav_funcs(theory_data_dir)
    print "Loaded grav sim data"

    datafiles = bu.find_all_fnames(data_dir, ext=config.extensions['data'])

    datafiles = datafiles[file_inds[0]:file_inds[1]]
    print "Processing %i files..." % len(datafiles)

    if len(datafiles) == 0:
        print "Found no files in: ", data_dir
        quit()


    fildat = get_data_at_harms(datafiles, minsep=minsep, maxthrow=maxthrow, \
                               beadheight=beadheight, plotfilt=plotfilt, \
                               cantind=0, ax1='x', ax2='z', diag=diag, plottf=False, \
                               nharmonics=nharmonics, harms=harms, \
                               ext_cant_drive=True, ext_cant_ind=1, \
                               max_file_per_pos=max_file_per_pos)

    alphadat = find_alpha_vs_file(fildat, gfuncs, yukfuncs, lambdas, lims, \
                                 ignoreX=ignoreX, ignoreY=ignoreY, ignoreZ=ignoreZ)

    
    fits, outdat = fit_alpha_vs_alldim(alphadat, lambdas, minsep=minsep, \
                                       maxthrow=maxthrow, beadheight=beadheight, plot=True)
    

    #alphas_bf, alphas_95cl, fits = \
    #            fit_alpha_vs_sep_1height(alphadat, height, minsep=minsep, maxthrow=maxthrow, \
    #                                     beadheight=beadheight)





'''





fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)
if diag:
    fig2, ax2 = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

if not plot_just_current:
    ax.loglog(lambdas, alphas_bf, linewidth=2, label='Best fit alpha')
    ax.loglog(lambdas, alphas_95cl, linewidth=2, label='95% CL')

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

'''
