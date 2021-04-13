import sys, re, os

import dill as pickle

import numpy as np
import pandas as pd

import scipy.interpolate as interpolate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")


ncore = 30
# ncore = 20
# ncore = 10
# ncore = 1


theory_base = '/home/cblakemore/opt_lev_analysis/gravity_sim/results/'
theory_data_dir = os.path.join(theory_base, '7_6um-gbead_1um-unit-cells_master/')


# data_dirs = ['/data/old_trap/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
#              '/data/old_trap/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz_elec-term', \
#              #\
#              '/data/old_trap/20180704/bead1/grav_data/shield', \
#              '/data/old_trap/20180704/bead1/grav_data/shield_1s_1h', \
#              #'/data/old_trap/20180704/bead1/grav_data/shield2', \
#              #'/data/old_trap/20180704/bead1/grav_data/shield3', \
#              #'/data/old_trap/20180704/bead1/grav_data/shield4', \
#              #'/data/old_trap/20180704/no_bead/grav_data/shield', \
#              #\
#              #'/data/old_trap/20180808/bead4/grav_data/shield1'
#              ]



# data_dirs = ['/data/new_trap/20191204/Bead1/Shaking/Shaking370/']
# data_dirs = ['/data/new_trap/20200107/Bead3/Shaking/Shaking380/']
# data_dirs = ['/data/new_trap/20200113/Bead1/Shaking/Shaking377/']
# data_dirs = [#'/data/new_trap/20200210/Bead2/Shaking/Shaking382/', \
#              '/data/new_trap/20200210/Bead2/Shaking/Shaking384/']

arg1 = str(sys.argv[1])
arg2 = str(sys.argv[2])
arg3 = int(sys.argv[3])

# data_dirs = ['/data/new_trap/20200320/Bead1/Shaking/Shaking373/']
data_dirs = ['/data/new_trap/20200320/Bead1/Shaking/Shaking378/']
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/Batch3/{:s}/'.format(arg)]
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/SBiN_2a/{:s}/'.format(arg)]
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/bkg_simple/{:s}/'.format(arg)]
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/StBiN/{:s}/'.format(arg1)]
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/StBiN2/{:s}/'.format(arg1)]
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/StBiN3/{:s}/'.format(arg1)]
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/raw/noise/']
# data_dirs = ['/data/new_trap_processed/mockfiles/20200320/output/noise/chas_tests/77/']
new_trap = True


signal_injection_path = ''
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_batch3_discovery_unc_3.p'
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_batch3_no-sig_discovery_3.p'
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_batch3_conservative_3.p'
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_sbin_2a_discovery.p'
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_bkg_simple_discovery.p'
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_stbin2_discovery.p'
# signal_injection_path = '/home/cblakemore/tmp/signal_injection_stbin3_discovery.p'
try:
    signal_injection_results = pickle.load(open(signal_injection_path, 'rb'))
except:
    signal_injection_results = {}

inj_key = arg1


binning_result_path = ''
# binning_result_path = '/home/cblakemore/tmp/20200320_mod_grav_binning.p'
# binning_result_path = '/home/cblakemore/tmp/20200320_mod_grav_rand{:d}_binning.p'.format(arg3)
# binning_result_path = '/home/cblakemore/tmp/20200320_mod_grav_far_binning.p'
# binning_result_path = '/home/cblakemore/tmp/20200320_mod_grav_far_rand{:d}_binning.p'.format(arg3)
# binning_result_path = '/home/cblakemore/tmp/signal_injection_stbin2_{:s}_binning.p'.format(arg1)
# binning_result_path = '/home/cblakemore/tmp/signal_injection_stbin2_{:s}_rand{:d}_binning.p'\
#                                     .format(arg1, arg3)
try:
    binning_results = pickle.load(open(binning_result_path, 'rb'))
except:
    binning_results = {}

bin_key = arg2


# step_cal_drive_freq = 41.0
step_cal_drive_freq = 71.0

pardirs_in_name = 1
# pardirs_in_name = 2

# substr = ''
# substr = 'Noise_add_3'
# substr = 'NoShaking_1'
# substr = 'Noise_batch'
# substr = 'Shaking0' # for 20200210/.../...382/
substr = 'Shaking3'  # for 20200210/.../...384/ and 20200320/.../...378
# substr = 'Shaking4'  # for 20200320/.../...373

user_load_ext = '_discovery'
# user_load_ext = '_no-discovery'

# user_save_ext = '_discovery'
# user_save_ext = '_no-discovery'
# user_save_ext = '_no-discovery_sign-sum'
user_save_ext = '_no-discovery_binning-{:s}'.format(arg2)
# user_save_ext = '_no-discovery_rand{:d}_binning-{:s}'.format(arg3, arg2)
# user_save_ext = '_no-discovery-conservative'
# user_save_ext = '_TEST'

# Nfiles = 5
# Nfiles = 50
# Nfiles = 1000
# Nfiles = 5000
# Nfiles = 5500  # for far 20200320 dataset
# Nfiles = 16000
Nfiles = 10000

suppress_off_diag = True

# reprocess = True
# save = True
reprocess = False
save = False

# redo_alpha_fit = True
redo_likelihood_sum = True
redo_alpha_fit = False
# redo_likelihood_sum = False

nalpha = 1001
# file_chunking = 5500
# file_chunking = 10000
file_chunking = int(arg2)
shuffle_in_time = True
if arg3 == 1:
    shuffle_seed = 123456      # rand1
elif arg3 == 2:
    shuffle_seed = 7654321     # rand2
elif arg3 == 3:
    shuffle_seed = 1029384756  # rand3
else:
    shuffle_seed = 999999
freq_pairing = 1
# freq_pairing = 8
# freq_pairing = 15
no_discovery = True
sum_by_sign = False
confidence_level = 0.95

plot_harms = False
plot_templates = False
plot_basis = False
plot_alpha_xyz = False
plot_bad_alphas = False

plot_mle_vs_time = False
mle_vs_time_chunk_size = 10
zoom_limits = ()
zoom_limits = (6.0, 6.5)
# plot_freqs = [6.0, 12.0, 33.0, 36.0]
plot_freqs = [12.0, 18.0, 21.0, 33.0, 36.0, 39.0]
plot_alpha = 1.0

plot_chunked_mle_vs_time = True
plot_mle_histograms = False
plot_likelihood_ratio_histograms = False
plot_harmonic_likelihoods = False
plot_final_likelihood = True
plot_limit = True

export_limit = True
export_path = '/data/new_trap_processed/limits/20200320_limit_binning-{:d}.p'\
                    .format(file_chunking)

lambdas_to_plot =  [10.0e-6]
# lambdas_to_plot = [5.0e-6, 10.0e-6]
# lambdas_to_plot = [5.0e-6, 10.0e-6, 12.0e-6, 18.0e-6, 31.0e-6]

limit_xlim = (5.0e-7, 1e-3)
limit_ylim = (5e6, 1e14)

save_hists = False

### Position of bead relative to the attractor coordinate system
p0_bead_dict = {'20200320': [392.4, 199.8, 42.37]}

# harms = [6]
# harms = [3,4,5,6]
harms = [2,4,6,7,10,11,12,13]
# harms = [2,3,4,5,6,7,8,9,10,11,12,13]
# harms = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# harms = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# harms = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30] # no 60 Hz
# harms = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,22,23,24,25,26,27,28,29,30] # no 51/60/63 Hz
# harms = [2,3,4,5,6,7,8,9,10]
# harms = []

### First level of keys for dict are {drive}{resp} pairs
### with axes indexed as x=0, y=1, z=2. So adjusting z phase
### response to a z drive would need the key '22'
adjust_phase = True
adjust_phase_dict  = {
                      '22': {3.0: 1.378, 6.0: 1.018, 9.0: 0.765, \
                             12.0: 0.657, 15.0: 0.455, 18.0: 0.344, \
                             21.0: 0.513, 24.0: 0.330, 27.0: 0.380, \
                             30.0: 0.626, 33.0: 0.681, 36.0: 0.620, \
                             39.0: 0.452}
                     }


fake_attractor_data = False
fake_attractor_data_freq = 3.0
fake_attractor_data_amp = 0.5*202.11
fake_attractor_data_dc = 194.92
fake_attractor_data_axis = 1

fix_sep = False
# fix_sep_val = 11.1
fix_sep_val = 13.9
# fix_sep_val = 19.9

fix_height = False
fix_height_val = -15.23



add_fake_data = False
fake_alpha = 5.0e10



######################################################################
######################################################################
if no_discovery:
    ss = True
else:
    ss = False


if plot_harms or plot_templates or plot_basis or plot_alpha_xyz:
    ncore = 1


#opt_ext = 'TEST'
opt_ext = '_harms'
for harm in harms:
    opt_ext += '-' + str(int(harm))

opt_ext += '_first-{:d}'.format(Nfiles)
if len(substr):
    opt_ext = '_{:s}{:s}'.format(substr, opt_ext)

if len(user_save_ext):
    opt_ext += user_save_ext


for ddir in data_dirs:
    # Skip the ones I've already calculated
    #if ddir == data_dirs[0]:
    #    continue
    print()

    aux_path_base = ddir.replace('/data/new_trap/', '/data/new_trap_processed/processed_files/')
    aux_path = os.path.join(aux_path_base, '{:s}_aux.pkl'.format(substr))
    try:
        aux_data = pickle.load( open(aux_path, 'rb') )
    except:
        print("Couldn't load auxiliary data file")
        aux_data = []


    paths = gu.build_paths(ddir, opt_ext, pardirs_in_name=pardirs_in_name, new_trap=new_trap)
    agg_path = paths['agg_path']
    plot_dir = paths['plot_dir']
    p0_bead = p0_bead_dict[paths['date']]

    agg_load_path = agg_path.replace(user_save_ext, user_load_ext)

    print('----------------------------------')
    if reprocess:
        print('Loading files from:')
        print('     {:s}'.format(ddir))
    else:
        print('Loading aggregate data from:')
        print('     {:s}'.format(agg_load_path))

    if save:
        print('----------------------------------')
        print('Will save to:')
        print('     {:s}'.format(agg_path))

    print('----------------------------------')
    print('Will save plots to:')
    print('     {:s}'.format(plot_dir))
    print('----------------------------------')
    print()

    if save:
        bu.make_all_pardirs(agg_path)


    if reprocess:

        datafiles, lengths = bu.find_all_fnames(ddir, ext=config.extensions['data'], \
                                                substr=substr, sort_by_index=True, \
                                                sort_time=False)
        datafiles = datafiles[:Nfiles]

        agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, \
                                   plot_harm_extraction=plot_harms, new_trap=new_trap, \
                                   step_cal_drive_freq=71.0, ncore=ncore, noisebins=10, \
                                   aux_data=aux_data, suppress_off_diag=suppress_off_diag, \
                                   fake_attractor_data=fake_attractor_data, \
                                   fake_attractor_data_amp=fake_attractor_data_amp, \
                                   fake_attractor_data_dc=fake_attractor_data_dc, \
                                   fake_attractor_data_freq=fake_attractor_data_freq, \
                                   fake_attractor_data_axis=fake_attractor_data_axis, \
                                   adjust_phase=adjust_phase, \
                                   adjust_phase_dict=adjust_phase_dict)

        agg_dat.load_grav_funcs(theory_data_dir)

        if save:
            agg_dat.save(agg_path)

        agg_dat.bin_rough_stage_positions()
        #agg_dat.average_resp_by_coordinate()

        # agg_dat.plot_force_plane(resp=0, fig_ind=1, show=True)
        # agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        # agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

        # agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
        #                                       ncore=ncore, plot_templates=plot_templates, \
        #                                       n_largest_harms=n_largest_harms, \
        #                                       # add_fake_data=True, fake_alpha=1e9,\
        #                                       )

        agg_dat.find_alpha_likelihoods_every_harm(plot=plot_alpha_xyz, plot_basis=plot_basis, \
                                                  ncore=ncore, plot_templates=plot_templates, \
                                                  add_fake_data=add_fake_data, \
                                                  fake_alpha=fake_alpha, fix_sep=fix_sep, \
                                                  fix_sep_val=fix_sep_val, fix_height=fix_height, \
                                                  fix_height_val=fix_height_val)

        if save:
            agg_dat.save(agg_path)

        agg_dat.sum_alpha_likelihoods(no_discovery=no_discovery, freq_pairing=freq_pairing, \
                                      nalpha=nalpha, chunk_size=file_chunking, \
                                      shuffle_in_time=shuffle_in_time, shuffle_seed=shuffle_seed, \
                                      sum_by_sign=sum_by_sign)
        if save:
            agg_dat.save(agg_path)


        print('Plotting/saving MLE histograms and profile likelihoods...', end='')
        sys.stdout.flush()

        if plot_mle_vs_time:
            agg_dat.plot_mle_vs_time(show=False, save=True, plot_freqs=plot_freqs, basepath=plot_dir, \
                                     plot_alpha=plot_alpha, chunk_size=mle_vs_time_chunk_size,  \
                                     zoom_limits=zoom_limits)

        if plot_chunked_mle_vs_time and no_discovery:
            agg_dat.plot_chunked_mle_vs_time(show=False, save=True, plot_freqs=plot_freqs, \
                                             basepath=plot_dir, plot_alpha=plot_alpha)

        if plot_mle_histograms:
            agg_dat.plot_mle_histograms(show=False, save=True, bins=20, basepath=plot_dir)

        if plot_likelihood_ratio_histograms:
            for lambda_to_plot in lambdas_to_plot:
                agg_dat.plot_likelihood_ratio_histograms(show=False, save=True, basepath=plot_dir, \
                                                         yuklambda=lambda_to_plot)

        if plot_harmonic_likelihoods:
            for lambda_to_plot in lambdas_to_plot:
                agg_dat.plot_sum_likelihood_by_harm(show=False, save=True, basepath=plot_dir, \
                                                    include_limit=True, no_discovery=no_discovery, \
                                                    confidence_level=confidence_level, ss=ss, \
                                                    yuklambda=lambda_to_plot)

        if plot_final_likelihood:
            for lambda_to_plot in lambdas_to_plot:
                agg_dat.plot_sum_likelihood(show=False, save=True, basepath=plot_dir, \
                                            include_limit=True, no_discovery=no_discovery, \
                                            confidence_level=confidence_level, ss=ss, \
                                            yuklambda=lambda_to_plot)

        if plot_limit:
            agg_dat.get_limit_from_likelihood_sum(confidence_level=confidence_level, \
                                                  no_discovery=no_discovery, ss=ss, \
                                                  xlim=limit_xlim, ylim=limit_ylim,
                                                  show=False, save=True, basepath=plot_dir, \
                                                  export_limit=export_limit, \
                                                  export_path=export_path)
        print('Done!')


        # agg_dat.fit_alpha_xyz_vs_alldim()
        # agg_dat.fit_alpha_xyz_onepos_simple(resp=[2], verbose=False)

        if save:
            agg_dat.save(agg_path)




    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms, new_trap=new_trap)
        agg_dat.load(agg_load_path)

        agg_dat.bin_rough_stage_positions()
        #agg_dat.average_resp_by_coordinate()

        if redo_alpha_fit:   
            # agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
            #                                         ncore=ncore, plot_bad_alphas=plot_bad_alphas, \
            #                                         plot_templates=plot_templates, \
            #                                         n_largest_harms=n_largest_harms, \
            #                                         # add_fake_data=True, fake_alpha=1e9, \
            #                                         )

            agg_dat.find_alpha_likelihoods_every_harm(plot=plot_alpha_xyz, plot_basis=plot_basis, \
                                                      ncore=ncore, plot_templates=plot_templates, \
                                                      add_fake_data=add_fake_data, \
                                                      fake_alpha=fake_alpha, fix_sep=fix_sep, \
                                                      fix_sep_val=fix_sep_val, \
                                                      fix_height=fix_height, \
                                                      fix_height_val=fix_height_val)
            if save:
                agg_dat.save(agg_path)

        # agg_dat.gfuncs_class.reload_grav_funcs()
        # agg_dat.save(agg_path)

        if redo_likelihood_sum:
            agg_dat.sum_alpha_likelihoods(no_discovery=no_discovery, freq_pairing=freq_pairing, \
                                          nalpha=nalpha, chunk_size=file_chunking, \
                                          shuffle_in_time=shuffle_in_time, shuffle_seed=shuffle_seed, \
                                          sum_by_sign=sum_by_sign)
            if save:
                agg_dat.save(agg_path)

        print('Plotting/saving MLE histograms and profile likelihoods...', end='')
        sys.stdout.flush()

        if plot_mle_vs_time:
            agg_dat.plot_mle_vs_time(show=False, save=True, plot_freqs=plot_freqs, basepath=plot_dir, \
                                     plot_alpha=plot_alpha, chunk_size=mle_vs_time_chunk_size, \
                                     zoom_limits=zoom_limits)
            
        if plot_chunked_mle_vs_time and no_discovery:
            agg_dat.plot_chunked_mle_vs_time(show=False, save=True, plot_freqs=plot_freqs, \
                                             basepath=plot_dir, plot_alpha=plot_alpha, \
                                             derp_key='rand{:d}'.format(arg3), \
                                             derp_nchunk=int(round(Nfiles/file_chunking)))

        if plot_mle_histograms:
            agg_dat.plot_mle_histograms(show=False, save=True, bins=20, basepath=plot_dir)

        if plot_likelihood_ratio_histograms:
            for lambda_to_plot in lambdas_to_plot:
                agg_dat.plot_likelihood_ratio_histograms(show=False, save=True, basepath=plot_dir, \
                                                         yuklambda=lambda_to_plot)

        if plot_harmonic_likelihoods:
            for lambda_to_plot in lambdas_to_plot:
                agg_dat.plot_sum_likelihood_by_harm(show=False, save=True, basepath=plot_dir, \
                                                    include_limit=True, no_discovery=no_discovery, \
                                                    confidence_level=confidence_level, ss=ss, \
                                                    yuklambda=lambda_to_plot)

        if plot_final_likelihood:
            for lambda_to_plot in lambdas_to_plot:
                agg_dat.plot_sum_likelihood(show=False, save=True, basepath=plot_dir, \
                                            include_limit=True, no_discovery=no_discovery, \
                                            confidence_level=confidence_level, ss=ss, \
                                            yuklambda=lambda_to_plot)
        if plot_limit:
            agg_dat.get_limit_from_likelihood_sum(confidence_level=confidence_level, \
                                                  no_discovery=no_discovery, ss=ss, \
                                                  xlim=limit_xlim, ylim=limit_ylim,
                                                  show=False, save=True, basepath=plot_dir, \
                                                  export_limit=export_limit, \
                                                  export_path=export_path)
        print('Done!')

        if save:
            agg_dat.save(agg_path)

        # agg_dat.fit_alpha_xyz_onepos_simple(resp=[2], verbose=False)

        #agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
        #agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        #agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

        # agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
        #                                         ncore=ncore)
        # agg_dat.plot_alpha_xyz_dict(k=0)
        # agg_dat.plot_alpha_xyz_dict(k=1)
        # agg_dat.plot_alpha_xyz_dict(k=2)
        # agg_dat.plot_alpha_xyz_dict(lambind=10)
        # agg_dat.plot_alpha_xyz_dict(lambind=50)



    # sample_lambdas = np.array([5.0e-6, 10.0e-6, 25.0e-6])

    if len(signal_injection_path) or len(binning_result_path):

        obj = agg_dat.agg_dict[list(agg_dat.agg_dict.keys())[0]][agg_dat.ax0vec[0]][agg_dat.ax1vec[0]][0]
        freqs = np.fft.rfftfreq(obj.nsamp, d=1.0/obj.fsamp)[obj.ginds]

        sample_lambdas = np.array([10.0e-6])
        # sample_lambdas = np.array([5.0, 10.0, 12.0, 18.0, 20.0, 25.0, 31.0]) * 1e-6

        mle_arr = np.zeros( (3,len(sample_lambdas),2) )
        mle_arr_2 = np.zeros( (3,len(sample_lambdas),len(freqs)) )

        limit_arr = np.zeros( (3,len(sample_lambdas),2) )
        limit_arr_2 = np.zeros( (3,len(sample_lambdas),len(freqs),2) )

        inds = []
        for yuklambda in sample_lambdas:
            inds.append( np.argmin( np.abs(yuklambda - agg_dat.pos_limit[0]) ) )
        inds = np.array(inds)

        for resp in [0,1,2]:

            func1 = interpolate.interp1d(np.log(agg_dat.pos_limit[0]), \
                                        np.log(agg_dat.pos_limit[resp+1]) )
            sample_posalphas = np.exp(func1(np.log(sample_lambdas)))

            # out_arr[resp,0] = sample_posalphas
            limit_arr[resp,:,0] = agg_dat.pos_limit[resp+1][inds]

            func2 = interpolate.interp1d(np.log(agg_dat.neg_limit[0]), \
                                        np.log(agg_dat.neg_limit[resp+1]) )
            sample_negalphas = np.exp(func2(np.log(sample_lambdas)))

            # out_arr[resp,1] = sample_negalphas
            limit_arr[resp,:,1] = agg_dat.neg_limit[resp+1][inds]


            mle_arr[resp,:,0] = agg_dat.mle[resp+1][inds]
            mle_arr[resp,:,1] = np.mean(agg_dat.mle_unc[resp,:,:][:,inds], axis=0)

            for freqind, freq in enumerate(freqs):
                harm_mles = agg_dat.mles_by_harmonic[freq]
                mle_arr_2[resp,:,freqind] = harm_mles[resp,inds,0]

                for i, ind in enumerate(inds):
                    prof_alpha, prof_val = agg_dat.likelihoods_sum_by_harmonic[freq][resp,ind]
                    limit = bu.get_limit_from_general_profile(prof_alpha, prof_val, ss=ss,\
                                                              no_discovery=no_discovery, \
                                                              confidence_level=confidence_level)
           
                    limit_arr_2[resp,i,freqind,0] = limit['upper_unc']
                    limit_arr_2[resp,i,freqind,1] = limit['lower_unc']

        if len(signal_injection_path):
            signal_injection_results['freqs'] = freqs
            signal_injection_results['sample_lambdas'] =  sample_lambdas
            signal_injection_results['key'] = 'MLE_array axes: coord-axis, sampled-lambda, (0)mle(1)unc\n'\
                    + 'MLE_by_harm axes: coord-axis, sampled-lambda, freq\n'\
                    + 'Limit axes: coord-axis, sampled-lambda, (0)pos-limit(1)neg-limit'

            signal_injection_results[inj_key+'_limit'] = limit_arr
            signal_injection_results[inj_key+'_limit_by_harm'] = limit_arr_2
            signal_injection_results[inj_key+'_mle'] = mle_arr
            signal_injection_results[inj_key+'_mle_by_harm'] = mle_arr_2
            pickle.dump(signal_injection_results, open(signal_injection_path, 'wb'))

        if len(binning_result_path):
            binning_results['freqs'] = freqs
            binning_results['sample_lambdas'] =  sample_lambdas
            binning_results['key'] = 'MLE_array axes: coord-axis, sampled-lambda, (0)mle(1)unc\n'\
                    + 'MLE_by_harm axes: coord-axis, sampled-lambda, freq\n'\
                    + 'Limit axes: coord-axis, sampled-lambda, (0)pos-limit(1)neg-limit'

            binning_results[bin_key+'_limit'] = limit_arr
            binning_results[bin_key+'_limit_by_harm'] = limit_arr_2
            binning_results[bin_key+'_mle'] = mle_arr
            binning_results[bin_key+'_mle_by_harm'] = mle_arr_2
            pickle.dump(binning_results, open(binning_result_path, 'wb'))