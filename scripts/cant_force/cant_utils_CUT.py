
        # Here we assume only one electrode of drive per file
        cind = np.unique(inds[0])[0]

        finds = inds[1] #frequency index with significant drive

        drive_freq_inds = []

        curr_ind = finds[0]
        temp_inds = []
        temp_inds.append(curr_ind)

        psd = dpsd[cind].flatten()

        for i in range(len(finds[1:])):
            ind = finds[i+1]
            if ind - curr_ind < 5.5:
                temp_inds.append(ind)
                curr_ind = ind
            else:
                if len(temp_inds) > 2:
                    temp_inds = np.array(temp_inds)
                    max_ind = np.argmax(psd[temp_inds]) + temp_inds[0]
                    drive_freq_inds.append(max_ind)
                curr_ind = ind
                temp_inds = []
                temp_inds.append(curr_ind)

        max_ind = np.argmax(psd[temp_inds]) + temp_inds[0]
        drive_freq_inds.append(max_ind)

        cinds = np.zeros(len(drive_freq_inds)) + cind

        voltage = self.electrode_settings[8 + cind]
        fac = (voltage / 0.004) * 1 # add in charge_step cal
        
        # Code left in a form easy to change to multiple freqs
        Hfreqs = []
        Hfreqs.append(self.electrode_settings[16 + cind])
        self.Hfreq = Hfreqs

        #b = finds>np.argmin(np.abs(self.fft_freqs - mfreq))

        Hout = Hmatst[:,:,drive_freq_inds]

        for i in range(len(drive_freq_inds)):
            ind = drive_freq_inds[i]
            Htemp = np.einsum('ijk->ij', Hmatst[:,:,[ind-1,ind,ind+1]]) / 3.
            Hout[:,:,i] = Htemp

        self.H = Hmat(drive_freq_inds, cinds, Hmatst[:,:,drive_freq_inds])
