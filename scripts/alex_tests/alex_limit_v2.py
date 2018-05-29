import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os
import glob
import matplotlib.mlab as ml
import sys
sys.path.append("microgravity")
import build_yukfuncs as yf
from scipy.optimize import minimize_scalar as ms
reload(bu)


#first define methods for extracting data from the df objects



class GravFile(bu.DataFile):
    '''class derived from bead util DataFile with special methods 
       for gravitational analysis'''
    
    def __init__(self, num_harmonics = 10):
        bu.DataFile.__init__(self)
        self.num_harmonics = num_harmonics
        self.freq_vector = "not computed"
        self.drive_freq = "not computed"
        self.drive_bin = "not compiuted"
        self.fft = "not computed"
        self.coef_at_harms = "not computed"
        self.amps_at_harms = "not computed"
        self.phis_at_harms = "not computed"
        self.sigma_phis_at_harms = "not computed"
        self.drive_phase = 0.
        self.noise = "not computed"
        self.harmonic_bins = "not computed"

    def get_freq_vector(self):
        '''generates frequency vector for np.fft.rfft'''
        n = np.shape(self.pos_data)[1]
        d = 1./self.fsamp
        self.freq_vector = np.fft.rfftfreq(n, d)

    def get_drive_freq(self, lbin = 10, get_from_data = True):
        '''determines cantilever drive frequency from attributes.'''
        if not get_from_data:
            if self.stage_settings['x driven']:
                self.drive_freq =  self.stage_settings['x freq']
            if self.stage_settings['y driven']:
                self.drive_freq = self.stage_settings['y freq']
            if self.stage_settings['z driven']:
                self.drive_freq = self.stage_settings['z freq']
        else:
            if type(self.freq_vector) == str:
                self.get_freq_vector()
            cfft = np.fft.rfft(self.cant_data)
            rms = np.sqrt(np.sum([np.abs(cfft[0, lbin:])**2, \
            np.abs(cfft[1, lbin:])**2, np.abs(cfft[2, lbin:])**2], axis = 1))
            driven_direction = np.argmax(rms)
            #print rms
            #print driven_direction
            dbin = np.argmax(np.abs(cfft[driven_direction, lbin:])) + lbin
            self.drive_phase = np.angle(cfft[driven_direction, dbin])
            self.drive_freq = self.freq_vector[dbin]

    def get_drive_bin(self):
        '''returns the fft bin corresponding to the cantilefer 
        drive frequency.'''
        if type(self.drive_freq) == str:
            self.get_drive_freq()
        if type(self.freq_vector) == str:
            self.get_freq_vector()
        self.drive_bin =  \
            np.argmin((self.freq_vector - self.drive_freq)**2)

    def get_harmonic_bins(self):
        '''computes vector of hrmonic bins'''
        if type(self.drive_bin) == str:
            self.get_drive_bin()
        self.harmonic_bins = self.drive_bin*\
                np.arange(1, self.num_harmonics + 1)
        

    def estimate_sig(self, hw = 20, use_diag = True, \
                     plot_psds = False, fplt_max = 999.):
        '''estimates  the coefficients at drive harmonics and uncertainty 
           from the amplitude of surrounding 2hW points.'''
        self.diagonalize()
        if type(self.harmonic_bins) == str:
            self.get_harmonic_bins()
        if (not use_diag)*(type(self.fft) == str):
            self.fft = np.fft.rfft(np.einsum('i, ij->ij', \
                    self.conv_facs, self.pos_data))
        elif type(self.fft) == str:
            self.fft = np.fft.rfft(self.diag_pos_data)
        
        self.noise = np.zeros((3, self.num_harmonics), dtype = complex)
        self.coef_at_harms = self.fft[:, self.harmonic_bins]

        for i, h in enumerate(self.harmonic_bins): 
            lfft = self.fft[:, h-hw:h]
            rfft = self.fft[:, h+1:h+hw+1]
            self.noise[:, i] = np.abs(np.median(np.concatenate(\
                               (lfft, rfft), axis = 1), axis = 1))
        self.noise = np.real(self.noise)
        self.amps_at_harms = np.real(np.abs(self.coef_at_harms))
        self.phis_at_harms = np.angle(self.coef_at_harms) - self.drive_phase 
        self.sigma_phis_at_harms = self.noise**2*self.amps_at_harms**2\
                                   /(2.*(self.amps_at_harms)**4)

        if plot_psds:
            max_bin = np.argmin((fplt_max - self.freq_vector)**2)
            f, axarr = plt.subplots(3,1, sharex = True, sharey = True)
            for i, ax in enumerate(axarr):
                norm = len(self.fft[i, :])
                ax.loglog(self.freq_vector[:max_bin], \
                        np.abs(self.fft[i, :max_bin])/norm,\
                        label = "Axis=" + str(i))
                ax.loglog(self.freq_vector[self.harmonic_bins], \
                        np.abs(self.fft[i, self.harmonic_bins])/norm, 'o', ms = 5)
            plt.xlabel("frequency [Hz]")
            plt.ylabel("Force [N]")
            plt.show()

    def plot_sig(self, just_amp = True):
        '''plots the result of estimate_sig. Calibrates into N by normalizing
           fft by 2/Nsamp'''
        if type(self.coef_at_harms) == str:
            self.estimate_sig()
        f, axarr = plt.subplots(3, 1, sharex = True, sharey = True)
        freqs = self.freq_vector[self.harmonic_bins]
        norm = len(self.freq_vector)
        for i, ax in enumerate(axarr):
            if just_amp:
                ax.errorbar(freqs, np.abs(self.coef_at_harms[i, :])/norm, \
                        self.noise[i, :]/(norm), fmt = 'o', ms = 7,\
                        label = "Amplitude")
            else:
                ax.errorbar(freqs, np.real(self.coef_at_harms[i, :])/norm, \
                        self.noise[i, :]/(np.sqrt(2)*norm), fmt = 'o', ms = 7,\
                        label = "Real")

                ax.errorbar(freqs, np.imag(self.coef_at_harms[i, :])/norm, \
                        self.noise[i, :]/(np.sqrt(2)*norm), fmt = 'o', ms = 7,\
                        label = "Imaginary")
        plt.legend()
        plt.xlabel("frequency [Hz]")
        plt.ylabel("Force [N]")
        return f, axarr

    def generate_template(self, yukfuncs_at_lam, p0):
        '''given a data file generates a template of the expected 
           force in the time domain.'''
        #first get cantilever position vector in same coord system as 
        #numerical integration of attractor mass
        if type(self.harmonic_bins) == str:
            self.get_harmonic_bins()
        pvec = np.zeros_like(self.cant_data)
        self.calibrate_stage_position()
        pvec[0, :] = self.cant_data[0, :] - p0[0]
        pvec[1, :] = self.cant_data[1, :] - p0[1]
        pvec[2, :] = self.cant_data[2, :] - p0[2]
        pts = np.stack(pvec*1e-6, axis = -1) #calibrate to m for yukfunc
        template_x_fft = np.fft.rfft(yukfuncs_at_lam[0](pts))
        template_y_fft = np.fft.rfft(yukfuncs_at_lam[1](pts))
        template_z_fft = np.fft.rfft(yukfuncs_at_lam[2](pts))
        template_fft = np.array([\
                template_x_fft, template_y_fft, template_z_fft])
        return template_fft[:, self.harmonic_bins]

    def plot_template(self, template, alpha, f, axarr, just_amp = True):
        '''plots template scalled by alpha'''
        freqs = self.freq_vector[self.harmonic_bins]
        norm = len(self.freq_vector)
        for i, ax in enumerate(axarr):
            if just_amp:
                ax.plot(freqs, np.abs(template[i, :])/norm,'o', ms = 7,\
                        label = "Template Amplitude")
            else:
                ax.plot(freqs, np.real(template[i, :])/norm, 'o', ms = 7,\
                        label = "Template Real")

                ax.plot(freqs, np.imag(template[i, :])/norm, 'o', ms = 7,\
                        label = "Template Imaginary")
        plt.legend()
        plt.xlabel("frequency [Hz]")
        plt.ylabel("Force [N]")
        return f, axarr

    def fit_alpha(self, template, noise, signif = 1.92, alpha_scale = 1e10, \
        fake_data = False, fake_alpha = 10**10, plot_profile = False):
        '''Chis square minimized fitting of data to scale of template.
           Returns '''
        if type(self.coef_at_harms) == str:
            self.estimate_sig()
        s = np.shape(self.coef_at_harms)
        ndof = 2*s[0]*s[1] #factor of 2 for complex numbers
        s = np.shape(self.coef_at_harms)
        fake_signal = fake_alpha*template + \
                 (np.random.randn(s[0], s[1]) + \
                  1.j*np.random.randn(s[0], s[1]))*noise/np.sqrt(2)

        def chi_sq(alpha):
           if fake_data:
               signal = fake_signal
            
           else:
               signal = self.coef_at_harms

           chisquared = np.sum(np.abs(signal - \
                        alpha*template)**2/(0.5*noise**2))/(ndof-1) 
           #print chisquared
           return chisquared
        chi_sq_scaled = lambda b: chi_sq(b*alpha_scale)
        res0 = ms(chi_sq_scaled)
        def delt_chi_sq(b):
            return (chi_sq_scaled(b) - res0.fun - signif)**2

        res1 = ms(delt_chi_sq)
        alpha_min = res0.x*alpha_scale
        error = np.abs((res0.x - res1.x))*alpha_scale
        if not res0.success or not res1.success:
            print "Warning: cant find min alpha"

        if plot_profile:
            test_alphas = np.linspace(alpha_min - 2.*error,\
                    alpha_min + 2.*error, 1000)
            #test_alphas = np.linspace(-10, 10, 10000)
            chi_sqs = np.array(map(chi_sq, test_alphas))
            plt.plot(test_alphas, chi_sqs)
            plt.xlabel("alpha")
            plt.ylabel("RCS")
            plt.show()

        return alpha_min, error, chi_sq

    def loop_over_lambdas(self, yukfuncs, noise, p0):
        '''fits alpha over an  array of yukfuncs at different lambda'''
        n = len(yukfuncs[0, :])
        alpha_maxs = np.zeros(n)
        for i in range(n):
            template = self.generate_template(yukfuncs[:, i], p0)
            alpha_min, error, chi_sq = self.fit_alpha(template, noise)
            alpha_maxs[i] = np.max(np.abs([alpha_min-error, alpha_min+error]))
        return alpha_maxs

