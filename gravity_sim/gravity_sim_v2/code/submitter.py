#!/usr/bin/python

import os, subprocess
import time
import numpy as np
import pickle as pickle

maxsubs = 3000
num_in_q = 500

rbeads = [2.40e-6]
#rbeads = [4.8e-6]

seps = np.linspace(4.0e-6, 84.0e-6, 81)
heights = np.linspace(-10e-6, 10e-6, 21)

#seps = [5.0e-6]
#heights = [0.0e-6]

params = []
for rbead in rbeads:
    for sep in seps:
        for height in heights:
            params.append((rbead, sep, height))
nparams = len(params)

confirmpath = "/farmshare/user_data/cblakemo/submits/"

exe = '/farmshare/user_data/cblakemo/gravity_sim/code/modgrav.sh'

badparams = []

resources = '-N 1 -n 1 -c 1 '   # Nodes, tasks, cores
time_alloc = '-t 1:00:00 '
mail = '--mail-user=cblakemo@stanford.edu --mail-type=FAIL '

prefix = 'sbatch ' + resources + time_alloc + mail

def count_jobs():
    output = os.popen('squeue -u cblakemo').read()
    num = output.count('\n') - 1
    return num

def submit_next_job(params):
    try:
        command = prefix + exe + ' ' + str(params[0][0]) + ' ' + str(params[0][1]) + ' ' + str(params[0][2])
    
        os.system(command)

        pickle.dump([], open(confirmpath + str(params[0]) + '.p', 'wb'))

    except:
        print('Failed', end=' ')
        params.append(params[0])
        badparams.append(params[0])

    return params[1:]


ind = 0
while (len(params) > 0 and ind < maxsubs):
    num = count_jobs()
    if num >= num_in_q:
	time.sleep(600)
	continue
    params = submit_next_job(params)
    time.sleep(1)
    ind += 1

pickle.dump(badparams, open('/farmshare/user_data/cblakemo/sim_results/badparams.p', 'wb'))

