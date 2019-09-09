import os, fnmatch, sys

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


year2019 = True


plt.rcParams.update({'font.size': 14})

save_mass = True
print_res = True
plot = True

save_example = False
example_filename = '/home/charles/plots/weigh_beads/example_extrapolation.png'

try:
    allres_dict = pickle.load(open('./allres.p', 'rb'))
except:
    allres_dict = {}

try:
    overall_mass_dict = pickle.load(open('./overall_masses.p', 'rb'))
except:
    overall_mass_dict = {}

file_dict = {}

#arr = []
#arr.append('/data/20181119/bead1/mass_meas/neg_charge_2')
#arr.append('/data/20181119/bead1/mass_meas/pos_charge_1')
#file_dict['20181119'] = (False, arr)


arr = []
arr.append('/data/20181129/bead1/weigh/high_pressure_0.5Hz_4pp')
arr.append('/data/20181129/bead1/weigh/low_pressure_0.5Hz_4pp')
arr.append('/data/20181129/bead1/weigh/low_pressure_0.5Hz_4pp_pos-charge')
file_dict['20181129'] = (arr, 1, 4)

arr = []
arr.append('/data/20181130/bead2/weigh/high_pressure_0.5Hz_4pp')
arr.append('/data/20181130/bead2/weigh/low_pressure_0.5Hz_4pp_neg-charge')
arr.append('/data/20181130/bead2/weigh/low_pressure_0.5Hz_4pp_pos-charge')
file_dict['20181130'] = (arr, 1, 4)


# With chopper
#chopper = True
arr = []
arr.append(['/data/20181211/bead2/weigh/high_pressure_neg_0.3Hz_4pp',\
            '/data/20181211/bead2/weigh/high_pressure_neg_0.5Hz_4pp'])
arr.append('/data/20181211/bead2/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20181211/bead2/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20181211'] = (arr, 5, 7)

arr = []
arr.append('/data/20181213/bead1/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20181213/bead1/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20181213/bead1/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20181213'] = (arr, 5, 7)

arr = []
arr.append('/data/20181231/bead1/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20181231/bead1/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20181231/bead1/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20181231'] = (arr, 5, 7)

arr = []
#arr.append(['/data/20190104/bead1/weigh/high_pressure_neg_0.5Hz_4pp', \
#            '/data/20190104/bead1/weigh/high_pressure_neg_0.5Hz_4pp_wfb', \
#            '/data/20190104/bead1/weigh/pumpdown_neg_0.5Hz_4pp'])
arr.append('/data/20190104/bead1/weigh/high_pressure_neg_0.5Hz_4pp')
#arr.append('/data/20190104/bead1/weigh/high_pressure_neg_0.5Hz_4pp_wfb')
#arr.append('/data/20190104/bead1/weigh/pumpdown_neg_0.5Hz_4pp')
arr.append('/data/20190104/bead1/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190104/bead1/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20190104'] = (arr, 5, 7)

arr = []
arr.append(['/data/20190108/bead1/weigh/high_pressure_neg_0.5Hz_4pp_fb1e-5', \
            '/data/20190108/bead1/weigh/high_pressure_neg_0.5Hz_4pp_fb3e-5', \
            '/data/20190108/bead1/weigh/high_pressure_neg_0.5Hz_4pp_fb7e-5', \
            '/data/20190108/bead1/weigh/high_pressure_neg_0.5Hz_4pp_fb1e-4', \
            '/data/20190108/bead1/weigh/high_pressure_neg_0.5Hz_4pp_fb3e-4', \
            '/data/20190108/bead1/weigh/high_pressure_neg_0.5Hz_4pp_fb7e-4'])
arr.append(['/data/20190108/bead1/weigh/low_pressure_neg_0.5Hz_4pp_fb7e-4', \
            '/data/20190108/bead1/weigh/low_pressure_neg_0.5Hz_4pp_fb7e-4_later'] )
file_dict['20190108'] = (arr, 5, 7)

arr = []  
arr.append('/data/20190109/bead1/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190109/bead1/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append(['/data/20190109/bead1/weigh/low_pressure_pos_0.5Hz_4pp', \
            '/data/20190109/bead1/weigh/low_pressure_pos_0.5Hz_4pp_2', \
            '/data/20190109/bead1/weigh/low_pressure_pos_0.5Hz_4pp_later'] )
file_dict['20190109'] = (arr, 5, 7)

arr = []  
arr.append('/data/20190110/bead1/weigh2/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190110/bead1/weigh2/low_pressure_pos_0.5Hz_4pp')
file_dict['20190110'] = (arr, 5, 7)


arr = []  
arr.append('/data/20190114/bead1/weigh/high_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190114/bead1/weigh/low_pressure_neg_0.5Hz_4pp')
arr.append('/data/20190114/bead1/weigh/low_pressure_pos_0.5Hz_4pp')
file_dict['20190114'] = (arr, 5, 7)


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

#file_dict = {'20190124': (True, arr)}


arr = []  ### 
arr.append('/daq2/20190408/bead1/weigh/lowp_neg_150Vpp')
file_dict['20190408'] = (arr, 1, 0)



arr = []  ### 
arr.append('/daq2/20190514/bead1/weigh/lowp_neg_4Vpp')
file_dict['20190514'] = (arr, 1, 0)


arr = []  ### 
arr.append('/daq2/20190619/bead1/weigh/50kHz_8Vpp')
file_dict['20190619'] = (arr, 1, 0)


arr = []  ### 
arr.append('/daq2/20190626/bead1/weigh/lowp_neg_8Vpp')
file_dict['20190626'] = (arr, 1, 0)


# arr = []  ### -13e charge for these
# arr.append('/daq2/20190829/bead1/weigh/lowp_2Vpp_2')
# arr.append('/daq2/20190829/bead1/weigh/lowp_4Vpp_2')
# arr.append('/daq2/20190829/bead1/weigh/lowp_6Vpp_2')
# file_dict['20190829'] = (arr, 1, 0)


arr = []  ### 
arr.append('/daq2/20190829/bead2/weigh/lowp_neg_6Vpp')
arr.append('/daq2/20190829/bead2/weigh/lowp_neg_8Vpp')
file_dict['20190829'] = (arr, 1, 0)



arr = []  ### 
arr.append('/daq2/20190905/bead1/weigh/lowp_neg_4Vpp')
file_dict['20190905'] = (arr, 1, 0)


file_dict = {'201900905': (arr, 1, 0)}




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



mon_fac = 100 # Tabor amplifier monitor is 100:1

trans_gain = 100e3  # V/A
pd_gain = 0.25      # A/W

line_filter_trans = 0.45
bs_fac = 0.01

maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

file_inds = (0, 500)

userNFFT = 2**12
diag = False

save = False

fullNFFT = False

###########################################################

def line(x, a, b):
    return a * x + b


def weigh_bead_efield(files, elec_ind, pow_ind, colormap='jet', sort='time',\
                      file_inds=(0,10000), plot=True, print_res=False, pos=False, \
                      save_mass=False):
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

    date = files[0].split('/')[2]

    charge_file = '/calibrations/charges/' + date
    if pos:
        charge_file += '_recharge.charge'
    else:
        charge_file += '.charge'

    try:
        q_bead = (np.load(charge_file)[0]) * constants.elementary_charge
    except:
        nq = raw_input('No charge file. Guess q: ')
        print int(nq)
        q_bead = int(nq) * constants.elementary_charge

    run_index = 0

    masses = []

    nfiles = len(files)
    if not print_res:
        print "Processing %i files..." % nfiles

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

        bu.progress_bar(fil_ind, nfiles)

        # Load data
        df = bu.DataFile()
        try:
            df.load(fil, load_other=True)
        except Exception:
            continue

        try:
            df.calibrate_stage_position()
            df.calibrate_phase()
        except:
            continue

        if ('20181129' in fil) and ('high' in fil):
            pressure_vec.append(1.5)
        else:
            pressure_vec.append(df.pressures['pirani'])

        freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
        drive_ind = np.argmin(np.abs(freqs-0.5))

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

        #plt.loglog(freqs, zfb)
        #print zfb[drive_ind]
        #print (np.sum(zfb) - zfb[drive_ind]) / (len(zfb))
        #plt.show()

        #badfile = (zfb[drive_ind]) < 500 * ((np.sum(zfb) - zfb[drive_ind]) / (len(zfb)))

        #if badfile:
        #    Nbad += 1
        #    continue


        top_elec = mon_fac * df.other_data[elec_ind]
        bot_elec = mon_fac * df.other_data[elec_ind+1]

        # plt.plot(top_elec)
        # plt.plot(bot_elec)
        # plt.show()

        #Vdiff = top_elec - bot_elec
        #eforce = -1.0 * (Vdiff / (4.0e-3)) * q_bead

        nsamp = len(top_elec)
        zeros = np.zeros(nsamp)

        voltages = [zeros, top_elec, bot_elec, zeros, \
                    zeros, zeros, zeros, zeros]
        efield = bu.trap_efield(voltages)
        eforce2 = efield[2] * q_bead

        #eforce2 = (top_elec * e_top_func(0.0) + bot_elec * e_bot_func(0.0)) * q_bead
        if noise:
            e_dc.append(np.mean(eforce2))
            e_ac_val = np.abs(np.fft.rfft(eforce2))[drive_ind]
            e_ac.append(e_ac_val * bu.fft_norm(df.nsamp, df.fsamp) \
                        * np.sqrt(freqs[1] - freqs[0]) )

        zphase_avg += (zphase[drive_ind] - np.angle(eforce2)[drive_ind])



        current = np.abs(df.other_data[pow_ind]) / trans_gain
            
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
        
        #plt.loglog(freqs, np.abs(fft1), label='Power')
        #plt.loglog(freqs, np.abs(fft2), label='FB')
        #plt.legend()
        #plt.figure()
        #plt.semilogx(freqs, np.angle(fft1) - np.angle(fft2))
        #plt.show()

        #plt.plot(Vdiff)
        #plt.figure()
        #plt.plot(power)
        #plt.show()

        #inds = np.abs(eforce) < 1e-13

        bins, dat, errs = bu.spatial_bin(eforce2, power, nbins=200, width=0.05, \
                                         dt=1.0/df.fsamp, harms=[1], \
                                         add_mean=True, verbose=False)

        #bins = eforce2
        #dat = power

        dat = dat / np.mean(dat)

        all_eforce.append(bins)
        all_power.append(dat)

        #plt.plot(bins, dat, 'o')
        #plt.show()

        popt, pcov = opti.curve_fit(line, bins*1.0e13, dat, \
                                    absolute_sigma=False, maxfev=10000)
        test_vals = np.linspace(np.min(eforce2*1.0e13), np.max(eforce2*1.0e13), 100)

        all_param.append(popt)

        fit = line(test_vals, *popt)

        lev_force = -popt[1] / (popt[0] * 1.0e13)

        #ulev_force = -(popt[1]+np.sqrt(pcov[1,1])) / ((popt[0]+np.sqrt(pcov[0,0])) * 1.0e13)
        #llev_force = -(popt[1]-np.sqrt(pcov[1,1])) / ((popt[0]-np.sqrt(pcov[0,0])) * 1.0e13)

        mass = lev_force / (9.806)

        #umass = ulev_force / 9.806
        #lmass = llev_force / 9.806

        mass_vec.append(mass)

    fit_inds = freqs <= 0.2
    fit_inds[0] = False

    data = powpsd[fit_inds]/Npsd
    fit_freqs = freqs[fit_inds]

    def func(a, b):
        return np.sum( (data - a * fit_freqs**b)**2 / data**2 )

    #m = iminuit.Minuit(func, print_level=0)
    #fmin, param = m.migrad()
    #print m.values

    #printpedN param[0]['value']
    #a = param[0]['value']
    #b = param[1]['value']

    #def newfunc(f):
    #    return a * f**b

    #drive_ind = np.argmax(powpsd[1:]) + 1
    #derp_freq = 1.0 / (2.0*60.0*60.0)

    #print newfunc(derp_freq) / (powpsd[drive_ind]/Npsd)

    #plt.loglog(freqs, powpsd/Npsd)
    #plt.loglog(fit_freqs, a * fit_freqs**b)
    #plt.show()


    if noise:
        print 'DC power: ', np.mean(p_dc), np.std(p_dc)
        print 'AC power: ', np.mean(p_ac), np.std(p_ac)
        print 'DC field: ', np.mean(e_dc), np.std(e_dc)
        print 'AC field: ', np.mean(e_ac), np.std(e_ac)
        return


    #plt.plot(mass_vec)

    mean_popt = np.mean(all_param, axis=0)

    mean_lev = np.mean(mass_vec) * 9.806
    plot_vec = np.linspace(np.min(all_eforce), mean_lev, 100)

    if plot:
        fig = plt.figure(dpi=200, figsize=(6,4))
        ax = fig.add_subplot(111)
        plt.plot(np.array(all_eforce).flatten()[::5]*1e12*(1.0/9.806)*1e3, \
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
        plt.plot(x_plotvec*1e12, yresid*100, 'o')
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
    final_err_stat = np.std(mass_vec) / np.sqrt(len(mass_vec))
    final_err_sys = np.sqrt((0.015**2 + 0.01**2) * final_mass**2)
    final_pressure = np.mean(pressure_vec)  

    if save_mass:
        suffix = ''
        if final_pressure > 1:
            suffix += '_highp'
        else:
            suffix += '_lowp'

        if q_bead < 0:
            suffix += '_neg'
        else:
            suffix += '_pos'

        save_arr = [final_mass, final_err_stat, final_err_sys]
        np.save(open('/calibrations/masses/'+date+suffix+'.mass', 'wb'), save_arr)

    print 'Bad Files: %i / %i' % (Nbad, nfiles)
    if print_res:
        gresid_fac = (2.0 * np.pi * freqs[drive_ind])**2 / 9.8

        print '      mass    [pg]: %0.1f' % (final_mass*1e15)
        print '      st.err  [pg]: %0.2f' % (final_err_stat*1e15)
        print '      sys.err [pg]: %0.2f' % (final_err_sys*1e15)
        print '      qbead    [e]: %i' % (q_bead/constants.elementary_charge)
        print '      P     [mbar]: %0.2e' % final_pressure
        print '      <P>    [arb]: %0.2e' % (power_avg / power_N)
        print '      zresid   [g]: %0.3e' % ((zamp_avg / zamp_N) * gresid_fac)
        print '      zphase [rad]: %0.3e' % (zphase_avg / zamp_N)
        print '      zfb    [arb]: %0.3e' % (zfb_avg / zfb_N)
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
        
    



dates = file_dict.keys()
dates.sort()

allres = []
overall_mass = []
for date in dates:
    print date
    allres_dict[date] = []
    data = file_dict[date]
    dirs = data[0]
    elec_ind = data[1]
    pow_ind = data[2]
    
    masses = []
    err_stat = []
    err_sys = []
    for cdir in dirs:
        print '  ', cdir

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
                                save_mass=save_mass)
        allres.append(dat)
        allres_dict[date].append(dat)
        masses.append(dat[0])
        err_stat.append(dat[1])
        err_sys.append(dat[2])
        
        #print allres
        print
    err_stat = np.array(err_stat)
    err_sys = np.array(err_sys)
    err_tot = np.sqrt(err_stat**2 + err_sys**2)
    overall_mass_vec = [np.average(masses, weights=1.0/err_stat**2), \
                        np.sqrt(1.0 / np.sum(1.0 / err_stat**2)), 
                        np.std(masses), np.mean(err_sys)]
    overall_mass.append( overall_mass_vec )
    overall_mass_dict[date] = overall_mass_vec
    allres.append(list(np.zeros_like(dat)))

    print
    print

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
