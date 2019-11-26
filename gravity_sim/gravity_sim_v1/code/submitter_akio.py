#!/usr/bin/python

import os
import time
import numpy as np

maxsubs = 3000

params = np.linspace(0,99,100)

exe = '/farmshare/user_data/akiok/RunForceCalcCC.exe'

def submit_next_job(params):
    try:
        command = 'srun --time=48:00:00' + ' ' + str(int(params[0]))
        os.system(command)
    except:
        print('Failed', end=' ')
        params.append(params[0])
    return params[1:]


ind = 0
while (len(params) > 0 and ind < maxsubs):
    #print len(params)
    params = submit_next_job(params)
    time.sleep(20)
    ind += 1

