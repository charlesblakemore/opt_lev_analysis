###############################################################################
#storing seemingly useful functions that no longer make sense where they are

def round_sig(x, sig=2):
    '''Round a number to a certain number of sig figs
           INPUTS: x, number to be rounded
                   sig, number of sig figs

           OUTPUTS: num, rounded number'''
    neg = False
    if x == 0:
        return 0
    else:
        if x < 0:
            neg = True
            x = -1.0 * x
        num = round(x, sig-int(math.floor(math.log10(x)))-1)
        if neg:
            return -1.0 * num
        else:
            return num



def thermal_psd_spec(f, A, f0, g):
    #The position power spectrum of a microsphere normalized so that A = (volts/meter)^2*2kb*t/M
    w = 2.*np.pi*f #Convert to angular frequency.
    w0 = 2.*np.pi*f0
    num = g * w0**2
    denom = ((w0**2 - w**2)**2 + w**2*g**2)
    return 2 * (A * num / denom) # Extra factor of 2 from single-sided PSD

def damped_osc_amp(f, A, f0, g):
    '''Fitting function for AMPLITUDE of a damped harmonic oscillator
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    denom = np.sqrt((w0**2 - w**2)**2 + w**2 * g**2)
    return A / denom


def damped_osc_phase(f, A, f0, g, phase0 = 0.):
    '''Fitting function for PHASE of a damped harmonic oscillator. 
       Includes an arbitrary DC phase to fit over out of phase responses 
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    return A * np.arctan2(-w * g, w0**2 - w**2) + phase0


def sum_3osc_amp(f, A1, f1, g1, A2, f2, g2, A3, f3, g3):
    '''Fitting function for AMPLITUDE of a sum of 3 damped harmonic oscillators.
           INPUTS: f [Hz], frequency 
                   A1,2,3, amplitude of the three oscillators
                   f1,2,3 [Hz], resonant frequency of the three oscs
                   g1,2,3 [Hz], damping factors

           OUTPUTS: Lorentzian amplitude of complex sum'''
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.abs(csum)



def sum_3osc_phase(f, A1, f1, g1, A2, f2, g2, A3, f3, g3, phase0=0.):
    '''Fitting function for PHASE of a sum of 3 damped harmonic oscillators.
       Includes an arbitrary DC phase to fit over out of phase responses
           INPUTS: f [Hz], frequency 
                   A1,2,3, amplitude of the three oscillators
                   f1,2,3 [Hz], resonant frequency of the three oscs
                   g1,2,3 [Hz], damping factors

           OUTPUTS: Lorentzian amplitude of complex sum'''
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.angle(csum) + phase0


def good_corr(drive, response, fsamp, fdrive):
    corr = np.zeros(int(fsamp/fdrive))
    response = np.append(response, np.zeros( int(fsamp/fdrive)-1 ))
    n_corr = len(drive)
    for i in range(len(corr)):
        #Correct for loss of points at end
        correct_fac = 2.0*n_corr/(n_corr-i) # x2 from empirical tests
        #correct_fac = 1.0*n_corr/(n_corr-i) # 
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac
    return corr


def corr_func(drive, response, fsamp, fdrive, good_pts = [], filt = False, band_width = 1):
    #gives the correlation over a cycle of drive between drive and response.

    #First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response  = response-np.mean(response)

    #bandpass filter around drive frequency if desired.
    if filt:
        b, a = sp.butter(3, [2.*(fdrive-band_width/2.)/fsamp, 2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = sp.filtfilt(b, a, drive)
        response = sp.filtfilt(b, a, response)

    #Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

    #Throw out bad points if desired
    if len(good_pts):
        response[-good_pts] = 0.
        lentrace = np.sum(good_pts)


    #corr_full = good_corr(drive, response, fsamp, fdrive)/(lentrace*drive_amp**2)
    corr_full = good_corr(drive, response, fsamp, fdrive)/(lentrace*drive_amp)
    return corr_full

def get_color_map( n ):
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=n)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

