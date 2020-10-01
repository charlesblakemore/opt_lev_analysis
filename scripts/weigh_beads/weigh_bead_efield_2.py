import os, fnmatch, sys, traceback, re

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.constants as constants

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config
import transfer_func_util as tf

import iminuit

plt.rcParams.update({'font.size': 14})

file_dict = {}





arr = []  ### FIRST BEAD ON ATTRACTOR
arr.append('/data/20190122/bead1/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190122/bead1/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append(['/data/20190122/bead1/weigh/low_pressure_pos_0.5Hz_4pp_2', \
            '/data/20190122/bead1/weigh/low_pressure_pos_0.5Hz_4pp_3'])
file_dict['20190122'] = (arr, 5, 7)



arr = []  ### SECOND BEAD ON ATTRACTOR
arr.append('/data/20190123/bead2/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190123/bead2/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190123/bead2/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20190123'] = (arr, 5, 7)


arr = []  ### THIRD BEAD ON ATTRACTOR
arr.append('/data/20190124/bead2/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190124/bead2/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190124/bead2/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20190124'] = (arr, 5, 7)





arr = []  ### 
arr.append('/data/old_trap/20200307/gbead1/weigh_2/4Vpp_lowp_0')
# arr.append('/data/old_trap/20200307/gbead1/weigh/6Vpp_lowp_0')
file_dict['20200307'] = (arr, 1, 0)


arr = []  ### 
arr.append('/data/old_trap/20200322/gbead1/weigh/4Vpp_neg')
arr.append('/data/old_trap/20200322/gbead1/weigh/6Vpp_neg')
arr.append('/data/old_trap/20200322/gbead1/weigh/8Vpp_neg')
file_dict['20200322'] = (arr, 1, 0)


arr = []  ### 
# arr.append('/data/old_trap/20200327/gbead1/weigh/4Vpp_neg_lowp')
# arr.append('/data/old_trap/20200327/gbead1/weigh/6Vpp_neg_lowp')
arr.append('/data/old_trap/20200327/gbead1/weigh/8Vpp_neg_lowp')
file_dict['20200327'] = (arr, 1, 0)


arr = []  ### 
# arr.append('/data/old_trap/20200330/gbead3/weigh/6Vpp_neg_lowp')
arr.append('/data/old_trap/20200330/gbead3/weigh/8Vpp_neg_lowp')
file_dict['20200330'] = (arr, 1, 0)


arr = []  ### 
arr.append('/data/old_trap/20200721/bead2/weigh/4Vpp_lowp_neg_1')
arr.append('/data/old_trap/20200721/bead2/weigh/6Vpp_lowp_neg_1')
file_dict['20200721'] = (arr, 1, 0)


arr = []  ### 
arr.append('/data/old_trap/20200727/bead1/weigh/4Vpp_lowp_neg')
arr.append('/data/old_trap/20200727/bead1/weigh/6Vpp_lowp_neg')
file_dict['20200727'] = (arr, 1, 0)


arr = []  ### 
arr.append('/data/old_trap/20200924/bead1/weigh/4Vpp_lowp_neg')
arr.append('/data/old_trap/20200924/bead1/weigh/6Vpp_lowp_neg')
file_dict['20200924'] = (arr, 2, 1)


file_dict = {'20200924': (arr, 2, 1)}





# arr = []  ### 
# arr.append('/data/new_trap/20200320/Bead1/Mass/derp')
# file_dict['20200320'] = (arr, 1, 0)


# file_dict = {'20200320': (arr, 1, 0)}



# Noise data
#chopper = True
noise = False
n_noise = 7

noise_dirs = ['/data/20181211/bead2/weigh/noise/no_charge_0.5Hz_4pp', \
              '/data/20181211/bead2/weigh/noise/no_drive', \
              '/data/20181213/bead1/weigh/noise/no_charge_0.5Hz_4pp', \
              '/data/20181213/bead1/weigh/noise/no-bead_0.5Hz_4pp', \
              '/data/20181213/bead1/weigh/noise/no-bead_0.5Hz_4pp_pd-blocked', \
              '/data/20181213/bead1/weigh/noise/no-bead_zfb-inject', \
              '/data/20181213/bead1/weigh/noise/no-bead_zfb-inject_pd-blocked']


# new_trap = True
new_trap = False

#r_divider = 50000.0 / (3000.0 + 50000.0)
r_divider = 1.0
mon_fac = r_divider**(-1) * 100.0 # Tabor amplifier monitor is 100:1

sign = -1.0
# sign = 1.0
trans_gain = 100e3  # V/A
pd_gain = 0.25      # A/W

# line_filter_trans = 0.45
line_filter_trans = 1
# bs_fac = 0.01
bs_fac = 0.09

maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

file_inds = (0, 500)

userNFFT = 2**12
diag = False

save = False

fullNFFT = False

correct_phase_shift = True

save_mass = True
print_res = True
plot = True

save_example = False
example_filename = '/home/cblakemore/plots/weigh_beads/example_extrapolation.png'

# upper_outlier = 95e-15  # in kg
upper_outlier = 95e-13

lower_outlier = 70e-15
# lower_outlier = 1e-15

try:
    allres_dict = pickle.load(open('./allres.p', 'rb'))
except:
    allres_dict = {}

try:
    overall_mass_dict = pickle.load(open('./overall_masses.p', 'rb'))
except:
    overall_mass_dict = {}

###########################################################

def line(x, a, b):
    return a * x + b


def weigh_bead_efield(files, elec_ind, pow_ind, colormap='plasma', sort='time',\
                      file_inds=(0,10000), plot=True, print_res=False, pos=False, \
                      save_mass=False, new_trap=False, correct_phase_shift=False):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: files, list of files names to extract data
               data_axes, list of pos_data axes to plot
               cant_axes, list of cant_data axes to plot
               elec_axes, list of electrode_data axes to plot
               diag, boolean specifying whether to diagonalize

       OUTPUTS: none, plots stuff
    '''
    date = re.search(r"\d{8,}", files[0])[0]
    suffix = files[0].split('/')[-2]

    if new_trap:
        trap_str = 'new_trap'
    else:
        trap_str = 'old_trap'

    charge_file = '/data/{:s}_processed/calibrations/charges/'.format(trap_str) + date
    save_filename = '/data/{:s}_processed/calibrations/masses/'.format(trap_str) \
                            + date + '_' + suffix + '.mass'
    bu.make_all_pardirs(save_filename)

    if pos:
        charge_file += '_recharge.charge'
    else:
        charge_file += '.charge'

    try:
        nq = np.load(charge_file)[0]
    except:
        user_nq = input('No charge file. Guess q: ')
        nq = int(user_nq)

    if correct_phase_shift:
        print('Correcting anomalous phase-shift during analysis.')

    # nq = -16
    print('qbead: {:d} e'.format(int(nq)))
    q_bead = nq * constants.elementary_charge   

    run_index = 0

    masses = []

    nfiles = len(files)
    if not print_res:
        print("Processing %i files..." % nfiles)

    all_eforce = []
    all_power = []

    all_param = []

    mass_vec = []
    
    p_ac = []
    p_dc = []

    e_ac = []
    e_dc = []

    pressure_vec = []

    zamp_avg = 0
    zphase_avg = 0
    zamp_N = 0
    zfb_avg = 0
    zfb_N = 0
    power_avg = 0
    power_N = 0

    Nbad = 0

    powpsd = []

    for fil_ind, fil in enumerate(files):# 15-65

        # 4
        # if fil_ind == 16 or fil_ind == 4:
        #     continue

        bu.progress_bar(fil_ind, nfiles)

        # Load data
        df = bu.DataFile()
        try:
            if new_trap:
                df.load_new(fil)
            else:
                df.load(fil, load_other=True)
        except Exception:
            traceback.print_exc()
            continue

        try:
            # df.calibrate_stage_position()
            df.calibrate_phase()
        except Exception:
            traceback.print_exc()
            continue

        if ('20181129' in fil) and ('high' in fil):
            pressure_vec.append(1.5)
        else:
            try:
                pressure_vec.append(df.pressures['pirani'])
            except Exception:
                pressure_vec.append(0.0)

        ### Extract electrode data
        if new_trap:
            top_elec = df.electrode_data[1]
            bot_elec = df.electrode_data[2]
        else:
            top_elec = mon_fac * df.other_data[elec_ind]
            bot_elec = mon_fac * df.other_data[elec_ind+1]

        fac = 1.0
        if np.std(top_elec) < 0.5 * np.std(bot_elec) \
                or np.std(bot_elec) < 0.5 * np.std(top_elec):
            print('Adjusting electric field since only one electrode was digitized.')
            fac = 2.0

        nsamp = len(top_elec)
        zeros = np.zeros(nsamp)

        voltages = [zeros, top_elec, bot_elec, zeros, \
                    zeros, zeros, zeros, zeros]
        efield = bu.trap_efield(voltages, new_trap=new_trap)
        eforce2 = fac * sign * efield[2] * q_bead


        tarr = np.arange(0, df.nsamp/df.fsamp, 1.0/df.fsamp)

        # fig, axarr = plt.subplots(2,1,sharex=True,figsize=(10,8))

        # axarr[0].plot(tarr, top_elec, label='Top elec.')
        # axarr[0].plot(tarr, bot_elec, label='Bottom elec.')
        # axarr[0].set_ylabel('Apparent Voltages [V]')
        # axarr[0].legend(fontsize=12, loc='upper right')

        # axarr[1].plot(tarr, efield[2])
        # axarr[1].set_xlabel('Time [s]')
        # axarr[1].set_ylabel('Apparent Electric Field [V/m]')

        # fig.tight_layout()

        # plt.show()
        # input()


        freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
        drive_ind = np.argmax(np.abs(np.fft.rfft(eforce2)))
        drive_freq = freqs[drive_ind]

        zamp = np.abs( np.fft.rfft(df.zcal) * bu.fft_norm(df.nsamp, df.fsamp) * \
                       np.sqrt(freqs[1] - freqs[0]) )
        zamp *= (1064.0e-9 / 2.0) * (1.0 / (2.9 * np.pi))
        zphase = np.angle( np.fft.rfft(df.zcal) )
        zamp_avg += zamp[drive_ind]
        zamp_N += 1

        #plt.loglog(freqs, zamp)
        #plt.scatter(freqs[drive_ind], zamp[drive_ind], s=10, color='r')
        #plt.show()


        zfb = np.abs(np.fft.rfft(df.pos_fb[2]) * bu.fft_norm(df.nsamp, df.fsamp) * \
                      np.sqrt(freqs[1] - freqs[0]) )
        zfb_avg += zfb[drive_ind]
        zfb_N += 1



        #eforce2 = (top_elec * e_top_func(0.0) + bot_elec * e_bot_func(0.0)) * q_bead
        if noise:
            e_dc.append(np.mean(eforce2))
            e_ac_val = np.abs(np.fft.rfft(eforce2))[drive_ind]
            e_ac.append(e_ac_val * bu.fft_norm(df.nsamp, df.fsamp) \
                        * np.sqrt(freqs[1] - freqs[0]) )

        zphase_avg += (zphase[drive_ind] - np.angle(eforce2)[drive_ind])


        if np.sum(df.power) == 0.0:
            current = np.abs(df.other_data[pow_ind]) / trans_gain
        else:
            fac = 1e-6
            current = fac * df.power / trans_gain
            
        power = current / pd_gain
        power = power / line_filter_trans
        power = power / bs_fac

        power_avg += np.mean(power)
        power_N += 1
        if noise:
            p_dc.append(np.mean(power))
            p_ac_val = np.abs(np.fft.rfft(power))[drive_ind]
            p_ac.append(p_ac_val * bu.fft_norm(df.nsamp, df.fsamp) \
                        * np.sqrt(freqs[1] - freqs[0]) )

        fft1 = np.fft.rfft(power)
        fft2 = np.fft.rfft(df.pos_fb[2])
        
        if not len(powpsd):
            powpsd = np.abs(fft1)
            Npsd = 1
        else:
            powpsd += np.abs(fft1)
            Npsd += 1

        # freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
        # plt.loglog(freqs, np.abs(np.fft.rfft(eforce2)))
        # plt.loglog(freqs, np.abs(np.fft.rfft(power)))
        # plt.show()
        # input()


        # fig, axarr = plt.subplots(2,1,sharex=True,figsize=(10,8))

        # axarr[0].plot(tarr, power)
        # axarr[0].set_ylabel('Measured Power [Arb.]')

        # axarr[1].plot(tarr, power)
        # axarr[1].set_xlabel('Time [s]')
        # axarr[1].set_ylabel('Measured Power [Arb.]')

        # bot, top = axarr[1].get_ylim()
        # axarr[1].set_ylim(1.05*bot, 0)

        # fig.tight_layout()

        # plt.show()
        # input()


        bins, dat, errs = bu.spatial_bin(eforce2, power, nbins=200, width=0.0, #width=0.05, \
                                         dt=1.0/df.fsamp, harms=[1], \
                                         add_mean=True, verbose=False, \
                                         correct_phase_shift=correct_phase_shift, \
                                         grad_sign=0)

        dat = dat / np.mean(dat)

        #plt.plot(bins, dat, 'o')
        #plt.show()

        popt, pcov = opti.curve_fit(line, bins*1.0e13, dat, \
                                    absolute_sigma=False, maxfev=10000)
        test_vals = np.linspace(np.min(eforce2*1.0e13), np.max(eforce2*1.0e13), 100)

        fit = line(test_vals, *popt)

        lev_force = -popt[1] / (popt[0] * 1.0e13)
        mass = lev_force / (9.806)

        #umass = ulev_force / 9.806
        #lmass = llev_force / 9.806

        if mass > upper_outlier or mass < lower_outlier:
            print('Crazy mass: {:0.2f} pg.... ignoring'.format(mass*1e15))
            # fig, axarr = plt.subplots(3,1,sharex=True)
            # axarr[0].plot(eforce2)
            # axarr[1].plot(power)
            # axarr[2].plot(df.pos_data[2])
            # ylims = axarr[1].get_ylim()
            # axarr[1].set_ylim(ylims[0], 0)
            # plt.show()
            continue

        all_param.append(popt)

        all_eforce.append(bins)
        all_power.append(dat)

        mass_vec.append(mass)

    if noise:
        print('DC power: ', np.mean(p_dc), np.std(p_dc))
        print('AC power: ', np.mean(p_ac), np.std(p_ac))
        print('DC field: ', np.mean(e_dc), np.std(e_dc))
        print('AC field: ', np.mean(e_ac), np.std(e_ac))
        return


    #plt.plot(mass_vec)

    mean_popt = np.mean(all_param, axis=0)

    mean_lev = np.mean(mass_vec) * 9.806
    plot_vec = np.linspace(np.min(all_eforce), mean_lev, 100)

    if plot:
        fig = plt.figure(dpi=200, figsize=(6,4))
        ax = fig.add_subplot(111)
        ### Plot force (in pN / g = pg) vs power
        plt.plot(np.array(all_eforce).flatten()[::5]*1e15*(1.0/9.806), \
                 np.array(all_power).flatten()[::5], \
                 'o', alpha = 0.5)
        #for params in all_param:
        #    plt.plot(plot_vec, line(plot_vec, params[0]*1e13, params[1]), \
        #             '--', color='r', lw=1, alpha=0.05)
        plt.plot(plot_vec*1e12*(1.0/9.806)*1e3, \
                 line(plot_vec, mean_popt[0]*1e13, mean_popt[1]), \
                 '--', color='k', lw=2, \
                 label='Implied mass: %0.1f pg' % (np.mean(mass_vec)*1e15))
        left, right = ax.get_xlim()
        # ax.set_xlim((left, 500))
        ax.set_xlim((left, 110))

        bot, top = ax.get_ylim()
        ax.set_ylim((0, top))
        
        plt.legend()
        plt.xlabel('Applied electrostatic force/$g$ (pg)')
        plt.ylabel('Optical power (arb. units)')
        plt.grid()
        plt.tight_layout()
        if save_example:
            fig.savefig(example_filename)
            fig.savefig(example_filename[:-4]+'.pdf')
            fig.savefig(example_filename[:-4]+'.svg')


        x_plotvec = np.array(all_eforce).flatten()
        y_plotvec = np.array(all_power).flatten()

        yresid = (y_plotvec - line(x_plotvec, mean_popt[0]*1e13, mean_popt[1])) / y_plotvec

        plt.figure(dpi=200, figsize=(3,2))
        plt.hist(yresid*100, bins=30)
        plt.legend()
        plt.xlabel('Resid. Power [%]')
        plt.ylabel('Counts')
        plt.grid()
        plt.tight_layout()


        plt.figure(dpi=200, figsize=(3,2))
        plt.plot(x_plotvec*1e15, yresid*100, 'o')
        plt.legend()
        plt.xlabel('E-Force [pN]')
        plt.ylabel('Resid. Pow. [%]')
        plt.grid()
        plt.tight_layout()



        derpfig = plt.figure(dpi=200, figsize=(3,2))
        #derpfig.patch.set_alpha(0.0)
        plt.hist(np.array(mass_vec)*1e15, bins=10)
        plt.xlabel('Mass (pg)')
        plt.ylabel('Count')
        plt.grid()
        #plt.title('Implied Masses, Each from 50s Integration')
        #plt.xlim(0.125, 0.131)
        plt.tight_layout()
        if save_example:
            derpfig.savefig(example_filename[:-4]+'_hist.png')
            derpfig.savefig(example_filename[:-4]+'_hist.pdf')
            derpfig.savefig(example_filename[:-4]+'_hist.svg')

        plt.show()

    

    final_mass = np.mean(mass_vec)
    final_err_stat = 0.5*np.std(mass_vec) #/ np.sqrt(len(mass_vec))
    final_err_sys = np.sqrt((0.015**2 + 0.01**2) * final_mass**2)
    final_pressure = np.mean(pressure_vec)  

    if save_mass:
        save_arr = [final_mass, final_err_stat, final_err_sys]
        np.save(open(save_filename, 'wb'), save_arr)

    print('Bad Files: %i / %i' % (Nbad, nfiles))
    if print_res:
        gresid_fac = (2.0 * np.pi * freqs[drive_ind])**2 / 9.8

        print('      mass    [pg]: %0.1f' % (final_mass*1e15))
        print('      st.err  [pg]: %0.2f' % (final_err_stat*1e15))
        print('      sys.err [pg]: %0.2f' % (final_err_sys*1e15))
        print('      qbead    [e]: %i' % (q_bead/constants.elementary_charge))
        print('      P     [mbar]: %0.2e' % final_pressure)
        print('      <P>    [arb]: %0.2e' % (power_avg / power_N))
        print('      zresid   [g]: %0.3e' % ((zamp_avg / zamp_N) * gresid_fac))
        print('      zphase [rad]: %0.3e' % (zphase_avg / zamp_N))
        print('      zfb    [arb]: %0.3e' % (zfb_avg / zfb_N))
        outarr = [ final_mass*1e15, final_err_stat*1e15, final_err_sys*1e15, \
                   q_bead/constants.elementary_charge, \
                   final_pressure, power_avg / power_N, \
                   (zamp_avg / zamp_N) * gresid_fac, \
                   zphase_avg / zamp_N, zfb_avg / zfb_N ]
        return outarr
    else:
        scaled_params = np.array(all_param)
        scaled_params[:,0] *= 1e13
    
        outdic = {'eforce': all_eforce, 'power': all_power, \
                  'linear_fit_params': scaled_params, \
                  'ext_masses': mass_vec}
    
        return outdic



#if noise:
#    for i in range(n_noise):
#        allfiles, lengths = bu.find_all_fnames(noise_dirs[i])
#        derpdat = weigh_bead_efield(allfiles, plot=True)
#        print 
#        print
        
    



dates = list(file_dict.keys())
dates.sort()

allres = []
overall_mass = []
for date in dates:
    print(date)
    allres_dict[date] = []
    data = file_dict[date]
    dirs = data[0]
    elec_ind = data[1]
    pow_ind = data[2]
    
    masses = []
    err_stat = []
    err_sys = []
    for cdir in dirs:
        print('  ', cdir)

        if type(cdir) == list:
            nametest = cdir[0]
        elif type(cdir) == str:
            nametest = cdir

        if 'pos' in nametest:
            pos = True
        else:
            pos = False

        allfiles, lengths = bu.find_all_fnames(cdir, sort_time=True, \
                                               verbose=False)
        dat = weigh_bead_efield(allfiles, elec_ind, pow_ind, pos=pos, \
                                print_res=print_res, plot=plot, \
                                save_mass=save_mass, new_trap=new_trap, \
                                correct_phase_shift=correct_phase_shift)
        allres.append(dat)
        allres_dict[date].append(dat)
        masses.append(dat[0])
        err_stat.append(dat[1])
        err_sys.append(dat[2])
        
        #print allres
        print()
    err_stat = np.array(err_stat)
    err_sys = np.array(err_sys)
    err_tot = np.sqrt(err_stat**2 + err_sys**2)
    overall_mass_vec = [np.average(masses, weights=1.0/err_stat**2), \
                        np.sqrt(1.0 / np.sum(1.0 / err_stat**2)), 
                        np.std(masses), np.mean(err_sys)]
    overall_mass.append( overall_mass_vec )
    overall_mass_dict[date] = overall_mass_vec
    allres.append(list(np.zeros_like(dat)))

    print()
    print()

allres = np.array(allres)
overall_mass = np.array(overall_mass)

if save:
    np.save('./allres.npy', allres)
    np.save('./overall_masses.npy', overall_mass)

    pickle.dump(allres_dict, open('./allres.p', 'wb'))
    pickle.dump(overall_mass_dict, open('./overall_masses.p', 'wb'))


'''
allfiles, lengths = bu.find_all_fnames(highp_n_dir, sort_time=True)
highp_n_dat = weigh_bead_efield(allfiles, plot=True)

allfiles, lengths = bu.find_all_fnames(lowp_n_dir, sort_time=True)
lowp_n_dat = weigh_bead_efield(allfiles, plot=True)

allfiles, lengths = bu.find_all_fnames(lowp_p_dir, sort_time=True)
lowp_p_dat = weigh_bead_efield(allfiles, plot=True, pos=True)
'''
