#!/usr/bin/python

import os
import subprocess
import time
import numpy as np
import pickle as pickle

maxsubs = 3000

rbeads = [2.40e-6]
#rbeads = [2.43e-6]
#rbeads = [4.8e-6]

seps = np.linspace(3.0e-6, 80.0e-6, 78)
heights = np.linspace(-10e-6, 10e-6, 21)

params = []
for rbead in rbeads:
    for sep in seps:
        for height in heights:
            params.append((rbead, sep, height))
nparams = len(params)

confirmpath = "/farmshare/user_data/cblakemo/submits/"

exe = '/farmshare/user_data/cblakemo/gravity_sim/code/save_force_curve_farmshare.py'
#exe = './save_force_curve_farmshare.py'
numinq = 100

badparams = []

def count_q():
    #counts my jobs
    output = subprocess.check_output(['qstat -u cblakemo'])
    tempn = output.count("\n")
    if tempn> 0:
        return tempn - 2
    else:
        return 0


def submit_next_job(params):
    try:
        command = 'sbatch --time=48:00:00' + ' ' + exe + ' ' + str(params[0][0]) + ' ' + str(params[0][1]) + ' ' + str(params[0][2])
        #print command
        #output = subprocess.check_output(['qsub -l h_rt=48:00:00', exe, str(params[0][0]), \
        #                                      str(params[0][1]), str(params[0][2]) ])
    
        os.system(command)
        #output = subprocess.check_output([command])    

        pickle.dump([], open(confirmpath + str(params[0]) + '.p', 'wb'))
        #print output

    except:
        print('Failed', end=' ')
        params.append(params[0])
        badparams.append(params[0])

    return params[1:]


ind = 0
while (len(params) > 0 and ind < maxsubs):
    #print len(params)
    params = submit_next_job(params)
    time.sleep(1)
    ind += 1

pickle.dump(badparams, open('/farmshare/user_data/cblakemo/sim_results/badparams.p', 'wb'))

'''

for initind in range(numinq):
    params = submit_next_job(params)
    time.sleep(2)

while True:
    
    currnum = count_q
    if currnum < numinq:
        params = submit_next_job(params)
        if len(params) == 0:
            break

    time.sleep(60)
''' 
