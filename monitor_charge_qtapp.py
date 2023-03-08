import os, time, sys

import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


import bead_util as bu


# dirname = '/data/20230306/bead2/discharge/'
data_directory = '/data/old_trap/20200727/bead1/discharge/fine/'

bu.configuration.col_labels['electrodes'] = [0, 1, 2, 3, 4, 5, 6, 7]

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}

ts = 10.0


################################################################
################################################################
################################################################


def process_file(fname):

    ### Try loading the datafile. If the load encounters an error
    ### (for instance if the DAQ has finished saving but the FPGA
    ### hasn't finished writing the file or something stupid) then
    ### catch all exceptions and try reloading again in 2 seconds
    df = bu.DataFile()
    try:
        df.load(fname)
    except:
        time.sleep(2)
        df.load(fname)

    ### Extract the drive and response
    drive = df.electrode_data[elec_ind]
    resp = df.pos_data[pos_ind]

    ### Compute the frequencies and FFTs, ignoring normalization
    ### for this qualitative monitoring script
    freqs = np.fft.rfftfreq(len(resp), d=1.0/df.fsamp)
    fft = np.fft.rfft(resp)
    dfft = np.fft.rfft(drive)

    ### Build ASD and phase spectrum arrays, mostly so the ASDs
    ### can be returned for eventual plotting
    amp = np.abs(fft)
    phase = np.angle(fft)

    damp = np.abs(dfft)
    dphase = np.angle(dfft)

    ### Find the drive frequency/index
    ind = np.argmax(damp[1:]) + 1
    drive_freq = freqs[ind]

    ### Compute the max correlation (phase in-sensitive)
    corr = amp[ind] / damp[ind]

    ### Compute the in-phase correlation
    inphase_corr = (corr * np.exp( 1.0j * (phase[ind] - dphase[ind]) )).real 

    return corr, inphase_corr, freqs, damp, amp



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        ### Make some handles to access the matplotlib figure and
        ### axes objects for plotting and labeling and re-sizing
        self.fig_handle = fig
        self.axes1 = fig.add_subplot(121)
        self.axes2 = fig.add_subplot(122)

        super(MplCanvas, self).__init__(fig)




class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, sleep_time, data_directory, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.data_directory = data_directory

        self.canvas = MplCanvas(self, width=10, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        ### Setup the empty correlation arrays
        self.corr = []
        self.inphase_corr = []

        ### Update the plot for the most recent acquisition
        self.mrf = ''
        self.update_plot()

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(sleep_time*1000))
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()



    def update_plot(self):

        if len(self.mrf) == 0:
            verbose = True
        else:
            verbose = False

        files, _ = bu.find_all_fnames(self.data_directory, \
                                      sort_time=True, \
                                      verbose=verbose)

        ### If the last file is new (or an empty string), then 
        ### process the most recent file and replot
        if (len(self.mrf) == 0) or (self.mrf != files[-1]):
            self.mrf = files[-1]

            ### Process the file
            corr, inphase_corr, freqs, damp, amp = \
                process_file(self.mrf)

            ### Add the data to the running record
            self.corr.append(corr)
            self.inphase_corr.append(inphase_corr)

            ### Plot the correlations
            self.canvas.axes1.cla()
            self.canvas.axes1.plot(self.corr, label='Max')
            self.canvas.axes1.plot(self.inphase_corr, label='In-phase')
            self.canvas.axes1.set_xlabel('File index')
            self.canvas.axes1.set_ylabel('Correlation [arb]')
            self.canvas.axes1.legend(fontsize=10, loc='upper right')

            ### Plot the drive and response spectra as a qualitative 
            ### handle on bead performance
            self.canvas.axes2.cla()
            self.canvas.axes2.loglog(freqs, damp, label='Drive')
            self.canvas.axes2.loglog(freqs, amp, label='Response')
            self.canvas.axes2.set_xlabel('Frequency [Hz]')
            self.canvas.axes2.set_ylabel('ASD [arb]')
            self.canvas.axes2.legend(fontsize=10, loc='upper right')

            self.canvas.fig_handle.tight_layout()

        self.canvas.draw()

### Run the pyqt stuff
app = QtWidgets.QApplication(sys.argv)
w = MainWindow(ts, data_directory)
app.exec_()


