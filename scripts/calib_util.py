import numpy as np
import bead_util as bu



file_path = ''

df = bu.DataFile()
df.load(file_path)

df.cant_data





def correlation(drive, response, fsamp, fdrive, filt = False, band_width = 1):
    '''Compute the full correlation between drive and response,
       correctly normalized for use in step-calibration.

       INPUTS:   drive, drive signal as a function of time
                 response, resposne signal as a function of time
                 fsamp, sampling frequency
                 fdrive, predetermined drive frequency
                 filt, boolean switch for bandpass filtering
                 band_width, bandwidth in [Hz] of filter

       OUTPUTS:  corr_full, full and correctly normalized correlation'''

    # First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response  = response-np.mean(response)

    # bandpass filter around drive frequency if desired.
    if filt:
        b, a = sp.butter(3, [2.*(fdrive-band_width/2.)/fsamp, \
                             2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = sp.filtfilt(b, a, drive)
        response = sp.filtfilt(b, a, response)
    
    # Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

    # Define the correlation vector which will be populated later
    corr = np.zeros(int(fsamp/fdrive))

    # Zero-pad the response
    response = np.append(response, np.zeros(int(fsamp / fdrive) - 1) )

    # Build the correlation
    n_corr = len(drive)
    for i in range(len(corr)):
        # Correct for loss of points at end
        correct_fac = 2.0*n_corr/(n_corr-i) # x2 from empirical test
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac

    return corr







def find_step_cal_response(file_obj, mfreq = 1.):
    '''Analyze a data step-calibraiton data file, find the drive frequency,
       correlate the response to the drive

       INPUTS:   file_obj, input file object
                 dpsd_thresh, arbitrary threshold

       OUTPUTS:  corr_full, full and correctly normalized correlation'''

    if type(self.data_fft) == str:
        self.get_fft()        

    drive = self.electrode_data[ecol]
    response = self.pos_data[pcol]

    N = len(self.pos_data[0])
    dt = 1. / self.Fsamp
    t = np.linspace(0,(N+cut_samp-1)*dt, N+cut_samp)
    t = t[cut_samp:]

    b, a = sig.butter(3, [2.*(drive_freq-band_width/2.)/self.Fsamp, \
                          2.*(drive_freq+band_width/2.)/self.Fsamp ], btype = 'bandpass')
    responsefilt = sig.filtfilt(b, a, response)

    ### CORR_FUNC TESTING ###
    #test = 3.14159 * np.sin(2 * np.pi * drive_freq * t)
    #test_corr = bu.corr_func(7 * drive, test, self.Fsamp, drive_freq)
    #print np.sqrt(2) * np.std(test)
    #print np.max(test_corr)
    #########################

    corr_full = bu.corr_func(drive, response, self.Fsamp, drive_freq)

    response_amp2 = np.max(corr_full)
    #response_amp2 = corr_full[0]

    drive_amp = np.sqrt(2) * np.std(drive)
    response_amp = np.sqrt(2) * np.std(responsefilt)

    sign = 1 #np.sign(np.mean(drive*responsefilt))

    self.step_cal_response = sign * response_amp2 / drive_amp






def build_step_cal_vec(self, drive_freq = 41., pcol = 0, ecol = 3, files=[0,1000]):
        # Generates an array of step_cal values for the whole directory.
        # First check to make sure files are loaded and H is computed.
        if type(self.fobjs) == str: 
            self.load_dir(simple_loader)
        
        for fobj in self.fobjs:
            fobj.find_step_cal_response(drive_freq = drive_freq, \
                                        pcol = pcol, ecol = ecol)

        i = 0
        vec = []
        for fobj in self.fobjs:
            if i < files[0] or i > files[1]:
                i += 1
                continue
            vec.append(fobj.step_cal_response)
            if len(vec) >= 2:
                if np.abs(vec[-1]) > 10. * np.abs(vec[-2]):
                    vec[-1] = vec[-2]
            i += 1

        self.step_cal_vec = vec





def step_cal(self, n_phi = 20, plate_sep = 0.004, \
             drive_freq = 41., amp_gain = 1.):
    # Produce a conversion between voltage and force given a directory with single electron steps.
    # Check to see that Hs have been calculated.
    if type(self.step_cal_vec) == str:
        self.build_step_cal_vec(drive_freq = drive_freq)

    #phi = np.mean(np.angle(dir_obj.step_cal_vec[:n_phi])) #measure the phase angle from the first n_phi samples.
    #yfit =  np.abs(dir_obj.step_cal_vec)*np.cos(np.angle(dir_obj.step_cal_vec) - phi)

    yfit = np.abs(self.step_cal_vec)
    bvec = [yfit<10.*np.mean(yfit)] #exclude cray outliers
    yfit = yfit[bvec] 

    happy_with_fit = False

    while not happy_with_fit:
        plt.figure(1)
        plt.ion()
        plt.plot(yfit, 'o')
        plt.show()

        print "CHARGE STEP CALIBRATION"
        print "Enter guess at number of steps and charge at steps [[q1, q2, q3, ...], [x1, x2, x3, ...], vpq]"
        nstep = input(": ")

        #function for fit with volts per charge as only arg.
        def ffun(x, vpq, offset):
            qqs = vpq*np.array(nstep[0])
            offarr = np.zeros(len(x)) + offset
            #try:
            #    offarr = np.zeros(len(x))
            #    offarr[x>nstep[-1]] += offset
            #except TypeError:
            #    if x > nstep[-1]:
            #        offarr = offset
            #    else:
            #        offarr = 0
            return bu.multi_step_fun(x, qqs, nstep[1]) + offarr

        xfit = np.arange(len(self.step_cal_vec))
        xfit = xfit[bvec]

        #fit
        p0 = [nstep[2],0.02]#Initial guess for the fit
        popt, pcov = curve_fit(ffun, xfit, yfit, p0 = p0, xtol = 1e-12)

        fitobj = Fit(popt, pcov, ffun)#Store fit in object.

        newpopt = np.copy(popt)
        newpopt[1] = 0.0

        normfitobj = Fit(newpopt / popt[0], pcov / popt[0], ffun)

        plt.close(1)
        f, axarr = plt.subplots(2, sharex = True, \
                                gridspec_kw = {'height_ratios':[2,1]})#Plot fit
        normfitobj.plt_fit(xfit, (yfit - popt[1]) / popt[0], \
                           axarr[0], ylabel="Normalized Response [e]")
        normfitobj.plt_residuals(xfit, (yfit - popt[1]) / popt[0], axarr[1])
        plt.show()

        happy = raw_input("does the fit look good? (y/n): ")
        if happy == 'y':
            happy_with_fit = True
        elif happy == 'n':
            f.clf()
            continue
        else:
            f.clf()
            print 'that was a yes or no question... assuming you are unhappy'
            sys.stdout.flush()
            time.sleep(5)
            continue

    plt.ioff()

    #Determine force calibration.
    fitobj.popt = fitobj.popt * 1./(amp_gain*bu.e_charge/plate_sep)
    fitobj.errs = fitobj.errs * 1./(amp_gain*bu.e_charge/plate_sep)
    self.charge_step_calibration = fitobj
