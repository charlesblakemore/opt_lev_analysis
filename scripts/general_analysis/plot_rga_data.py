import sys, os, time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import rga_util as ru
import bead_util as bu

import dill as pickle

plt.rcParams.update({'font.size': 14})


#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/background2000001.txt'
#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/background_nextday000001.txt'

#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/preleak000001.txt'
#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/leak000001.txt'
#rga_data_file2 = '/daq2/20190514/bead1/rga_scans/postleak000001.txt'

#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/pre_Ar-leak_3_000001.txt'
#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/Ar-leak_2_000001.txt'
#rga_data_file2 = '/daq2/20190514/bead1/rga_scans/post_Ar-leak_4_000001.txt'

main_gases = ['He', 'N2', 'Ar', 'Kr', 'Xe', 'SF6']
#main_gases = ['SF6']
pramps = [1,2,3]
#pramps = [1]
# pramp_index = 3
# gas = 'Xe'

extract_dict = {'He': ['He', 'N2', 'H2O', 'O2'], \
                'N2': ['He', 'N2', 'H2O', 'O2', 'Ar'], \
                'Ar': ['He', 'N2', 'H2O', 'O2', 'Ar'], \
                'Kr': ['He', 'N2', 'H2O', 'O2', 'Ar', 'Kr'], \
                'Xe': ['He', 'N2', 'H2O', 'O2', 'Ar', 'Kr', 'Xe'], \
                'SF6': ['He', 'N2', 'H2O', 'O2', 'SF6']}

ion_dict = {'He': ['F+'], \
            'N2': ['F+'], \
            'Ar': ['H3O+', 'F+', 'Kr78+2', 'Kr80+2', 'Kr82+2'], \
            'Kr': ['H3O+', 'F+', 'Ar+'], \
            'Xe': ['H3O+', 'O2+2', 'F+'], \
            'SF6': ['H3O+', 'Ar+', 'Xe128+', 'Xe128+2', 'Xe129+', 'Xe129+2', \
                    'Xe130+', 'Xe130+2', 'O2+', 'O2+2']}

plot_extraction = True
remove_neg_diffs = True

load = False
save_fig = True
show = False
measurements = []
for gas in main_gases:
    for pramp_index in pramps:
        base = '/daq2/20190626/bead1/spinning/pramp/%s/rga/' % gas
        rga_data_file1 = base + 'before_%i_000001.txt' % pramp_index
        rga_data_file2 = base + 'flush_%i_000001.txt' % pramp_index

        save_base = '/processed_data/spinning/pramp_data/20190626/%s/' % gas
        save_file_before = save_base + '%s_pramp_%i_rga_before.p' % (gas, pramp_index + 0)
        save_file_flush = save_base + '%s_pramp_%i_rga_flush.p' % (gas, pramp_index + 0)

        fig_filename_base = '/home/charles/plots/20190626/pramp/%s/flush%i_' % (gas, pramp_index + 0)

        measurements.append([gas, rga_data_file1, rga_data_file2, \
                             save_file_before, save_file_flush, \
                             save_base, fig_filename_base, pramp_index])

        bu.make_all_pardirs(save_file_before)
        bu.make_all_pardirs(save_base)
        bu.make_all_pardirs(fig_filename_base)

#rga_data_file1 = base + 'He_20190607_measurement_2/meas2_He-leak_2_000001.txt'
#rga_data_file2 = base + 'He_20190607_measurement_2/meas2_He-leak_3_000001.txt'



# base1 = '/daq2/20190514/bead1/spinning/pramp3/He/rga_scans/'
# rga_data_file1 = base1 + 'He_20190607_measurement_1/meas1_pre-He-leak_2_000001.txt'

# base2 = '/daq2/20190625/rga_scans/'
# rga_data_file2 = base2 + 'reseat_with-grease_000002.txt'



plot_scan = False

plot_many_scans = False
plot_last_scans = True
last_nscans = 5
scan_ind = -10

plot_together = True

title = ''#'RGA Scan Evolution: Reseating Window'


#arrow_len = 0.02
#arrow_len = 0.2
arrow_len = 2


######################################################

outputs = []
for meas in measurements:
    gas = meas[0]
    rga_data_file1 = meas[1]
    rga_data_file2 = meas[2]
    save_file_before = meas[3]
    save_file_flush = meas[4]
    save_base = meas[5]
    fig_base =  meas[6]
    pramp_index = meas[7]
    gases_to_extract = extract_dict[gas]
    ions_to_ignore = ion_dict[gas]

    if not load:

        if gas == 'SF6':
            ions_to_ignore_before = ions_to_ignore + ['F+']
        else:
            ions_to_ignore_before = ions_to_ignore

        dat1 = ru.get_rga_data(rga_data_file1, many_scans=True, last_nscans=last_nscans, \
                               scan_ind=scan_ind, plot=plot_scan, plot_many=plot_many_scans, \
                               plot_last_scans=plot_last_scans, plot_nscans=last_nscans, \
                               gases_to_extract=gases_to_extract, ions_to_ignore=ions_to_ignore_before, \
                               plot_extraction=plot_extraction, save_fig=save_fig, show=show, \
                               fig_base=fig_base+'before_')

        pickle.dump(dat1, open(save_file_before, 'wb'))


        dat2 = ru.get_rga_data(rga_data_file2, many_scans=True, last_nscans=last_nscans, \
                               scan_ind=scan_ind, plot=plot_scan, plot_many=plot_many_scans, \
                               plot_last_scans=plot_last_scans, plot_nscans=last_nscans, \
                               gases_to_extract=gases_to_extract, ions_to_ignore=ions_to_ignore, \
                               plot_extraction=plot_extraction, save_fig=save_fig, show=show, \
                               fig_base=fig_base+'flush_')

        pickle.dump(dat2, open(save_file_flush, 'wb'))

    else:

        rga_data_file1 = meas[1]
        rga_data_file2 = meas[2]

        dat1 = pickle.load(open(save_file_before, 'rb'))
        dat2 = pickle.load(open(save_file_flush, 'rb'))

    outputs.append([dat1, dat2, save_base, fig_base, gas, pramp_index])


for output in outputs:
    dat1, dat2, save_base, fig_base, gas, pramp_index = output

    m1 = dat1['mass_vec']
    m2 = dat2['mass_vec']

    pp1 = dat1['partial_pressures']
    pp2 = dat2['partial_pressures']

    gas_pp1 = dat1['gas_pressures']
    gas_pp2 = dat2['gas_pressures']

    p1 = dat1['pressure']
    p2 = dat2['pressure']

    e1 = dat1['errs']
    e2 = dat2['errs']

    rga_diff = pp2 - pp1
    diff_err = np.sqrt(e1**2 + e2**2)

    pp_mean = 0.5 * (pp1 + pp2)
    p_mean = 0.5 * (dat1['pressure'] + dat2['pressure'])

    #p_tot = p_mean
    p_tot = dat1['pressure']


    if plot_together:

        fun = lambda x: '{:0.3g}'.format(x)
        title_str = 'Scan Comparison'

        fig, ax = plt.subplots(1,1,dpi=150,figsize=(10,3))
        ax.errorbar(m1, pp1, yerr=e1, color='C0')
        ax.fill_between(m2, pp1, np.ones_like(pp1)*1e-10,\
                        alpha=0.5, color='C0', label=('Before: ' + fun(p1) + ' mbar'))
        ax.errorbar(m2, pp2, yerr=e2, color='C1')
        ax.fill_between(m2, pp2, np.ones_like(pp2)*1e-10,\
                        alpha=0.5, color='C1', label=('After: ' + fun(p2) + ' mbar'))
        ax.set_ylim(1e-9, 2*np.max([np.max(pp1), np.max(pp2)]) )
        ax.set_xlim(0,int(np.max([np.max(m1), np.max(m2)])))
        ax.set_yscale('log')
        ax.set_xlabel('Mass [amu]')
        ax.set_ylabel('Partial Pressure [mbar]')
        if len(title_str):
            fig.suptitle(title_str)
        plt.tight_layout()
        plt.legend()
        if len(title_str):
            plt.subplots_adjust(top=0.87)

        if save_fig:
            fig.savefig(fig_base+'both-scans.png')

        if show:
            plt.show()

        plt.close(fig)



    fig, ax = plt.subplots(1,1,dpi=150,figsize=(10,3))
    ax.errorbar(m1, rga_diff/p_tot, yerr=diff_err/p_tot)
    #ax.errorbar(m1, diff/pp1, yerr=diff_err/p_tot)


    gases_to_label = ru.gases_to_label[gas]
    gas_keys = gases_to_label.keys()
    gas_keys.sort(key = lambda x: gases_to_label[x])
    labels = []
    neg = False
    negmax = 0
    pos = False
    posmax = 0
    last_val = 0
    last_mass = 0
    long_arrow = False
    for gas in gas_keys:

        max_val = np.max(rga_diff/p_tot)
        arrow_len = 0.25 * max_val

        mass = gases_to_label[gas]
        mass_ind = np.argmin( np.abs(m1 - mass) )

        val_init = rga_diff[mass_ind]
        if val_init > 0:
            pos = True
            val = np.max((rga_diff/p_tot)[mass_ind-5:mass_ind+5]) + \
                    np.max((diff_err/p_tot)[mass_ind-5:mass_ind+5])
            if val > posmax:
                posmax = val
        if val_init < 0:
            neg = True
            val = np.min((rga_diff/p_tot)[mass_ind-5:mass_ind+5]) - \
                    np.max((diff_err/p_tot)[mass_ind-5:mass_ind+5])
            if val < negmax:
                negmax = val

        if last_mass != 0:
            cond1 = np.abs(mass - last_mass) <= 3.5
            cond2 = np.sign(val) == np.sign(last_val)
            if cond1 and cond2:
                diff = np.abs(last_val - val)
                if diff < arrow_len:
                    if not long_arrow:
                        arrow_len *= 2
                        long_arrow = True
                    else:
                        arrow_len *= 3
            else:
                long_arrow = False

        offset = 0.025 * max_val

        labels.append(ax.annotate(gas, (mass, val + np.sign(val)*offset), \
                                  xytext=(mass, val + np.sign(val)*(arrow_len+offset)), \
                                  ha='center', va='center', \
                                  arrowprops={'width': 2, 'headwidth': 3, \
                                              'headlength': 5, 'shrink': 0.0}, \
                                  fontsize=10, zorder=100))
        labels[-1].set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w', pad=0))

        last_val = val
        last_mass = mass

    ax.set_xlabel('Mass [amu]')
    ax.set_ylabel('$(\Delta P \, / \, P_{\mathrm{init}})$ [abs]')
    ax.set_xlim(0,150)

    y1, y2 = ax.get_ylim()
    if pos:
        y2 = posmax + 2.0*arrow_len
    if neg:
        y1 = negmax - 2.0*arrow_len
    ax.set_ylim(y1, y2)

    if len(title):
        fig.suptitle(title)

    plt.tight_layout()
    if len(title):
        plt.subplots_adjust(top=0.89)

    if save_fig:
        fig.savefig(fig_base+'scan-diff.png')

    if show:
        plt.show()

    plt.close(fig)



    m0_arr = ru.get_leak_m0(gas, gas_pp1, gas_pp2, remove_neg_diffs=remove_neg_diffs)
    m0_filename = os.path.join(save_base, 'rga-m0_%i.mass' % pramp_index)

    np.save(open(m0_filename, 'wb'), m0_arr)

