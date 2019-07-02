import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu

plt.rcParams.update({'font.size': 14})

def extract_mass_vec(lines):

    for line_ind, line in enumerate(lines):
        if line[0:6] == '"Scan"':
            mass_line = line
            mass_line_ind = line_ind
            break

    nscans = len(lines) - 1 - mass_line_ind

    mass_strs = mass_line[7:].split('\t')
    mass_strs.pop(-1)
    mass_strs.pop(-1)
    extract = lambda x: float(x[6:-1])

    masses = np.array(map(extract, mass_strs))

    return {'mass_vec': masses, 'nscans': nscans, \
            'mass_line_ind': mass_line_ind}



def extract_scans(lines, mass_line_ind, nscans):

    def debug(str_in):
        print str_in
        return float(str_in)

    badscan = 0
    scans = []
    pressures = []
    for i in range(nscans):

        scan_line = lines[mass_line_ind + 1 + i]
        scan_strs = scan_line.split('\t')

        scan_strs.pop(0)
        scan_strs.pop(-1)

        if scan_strs[-1] == '':
            badscan += 1
            continue

        pp_arr = np.abs(np.array(map(float, scan_strs[:-1])))
        #pp_arr = np.abs(np.array(map(debug, scan_strs[:-1])))
        scans.append(pp_arr)

        try:
            pressures.append(float(scan_strs.pop(-1)))
        except:
            pressures.append(np.sum(pp_arr))

    scans = np.array(scans)

    return {'nscans': nscans-badscan, 'scans': scans, 'pressures': pressures}



def get_rga_data(rga_data_file, all_scans=True, scan_ind=0, plot=True, \
                 plot_many=False):

    print rga_data_file

    file_obj = open(rga_data_file)
    lines = file_obj.readlines()

    mass_data = extract_mass_vec(lines)
    mass_vec = mass_data['mass_vec']

    scan_data = extract_scans(lines, mass_data['mass_line_ind'], \
                              mass_data['nscans'])
    nscans = scan_data['nscans']
    print len(mass_vec)

    plot_x = mass_vec
    if all_scans:
        plot_y = np.mean(scan_data['scans'], axis=0)
        plot_errs = np.std(scan_data['scans'], axis=0)
        pressure = np.mean(scan_data['pressures'])
    else:
        plot_y = scan_data['scans'][scan_ind]
        plot_errs = np.zeros_like(plot_y) #* 0.1 * np.min(plot_y)
        pressure = scan_data['pressures'][scan_ind]

    if plot:
        title_str = 'Total Pressure: %0.3g torr' % pressure

        fig, ax = plt.subplots(1,1,dpi=150,figsize=(10,3))
        ax.errorbar(plot_x, plot_y, yerr=plot_errs)
        ax.fill_between(plot_x, plot_y, np.ones_like(plot_y)*1e-9,\
                        alpha=0.5)
        ax.set_ylim(1e-9, 2*np.max(plot_y))
        ax.set_xlim(0,int(np.max(plot_x)))
        ax.set_yscale('log')
        ax.set_xlabel('Mass [amu]')
        ax.set_ylabel('Partial Pressure [torr]')
        fig.suptitle(title_str)
        plt.tight_layout()
        plt.subplots_adjust(top=0.87)
        if not plot_many:
            plt.show()

    if plot_many:
        title_str = 'Total Pressure: %0.3g torr' % pressure

        colors = bu.get_color_map(nscans, cmap='inferno')
        fig2, ax2 = plt.subplots(1,1,dpi=150,figsize=(10,3))
        for i in range(nscans):
            ax2.plot(plot_x, scan_data['scans'][i], label=str(i), 
                    color=colors[i])
        ax2.set_ylim(1e-9, 2*np.max(plot_y))
        ax2.set_xlim(0,int(np.max(plot_x)))
        ax2.set_yscale('log')
        ax2.set_xlabel('Mass [amu]')
        ax2.set_ylabel('Partial Pressure [torr]')
        fig2.suptitle(title_str)
        plt.tight_layout()
        plt.subplots_adjust(top=0.87)
        plt.legend(ncol=2)
        plt.show()

    return {'mass_vec': plot_x, 'partial_pressures': plot_y, \
            'errs': plot_errs, 'pressure': pressure}

