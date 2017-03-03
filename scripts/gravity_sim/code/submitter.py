import os
import subprocess
import time

Tb = 1.
Ts = 1.
Tc = 1.04

dTc = 0.01
numperT = 480

outpath = "/afs/.ir.stanford.edu/users/a/r/arider/gas_sim_data/temp_diffs1"
if not os.path.exists(outpath):
    os.makedirs(outpath)
    print outpath


exe = './simfuncl.o'
numinq = 480

def count_q():
    #counts my jobs
    output = subprocess.check_output(['qstat'])
    tempn = output.count("\n")
    if tempn> 0:
        return tempn - 2
    else:
        return 0

def top_off_que(n, find_start, Tc, nTc):
    #adds jobs to the que to keep n in the que. returns the next file index.
    nq = count_q()
    for j in range(n - nq):
        outpathTc = outpath + '/Tc' + str(Tc)
        if not os.path.exists(outpathTc):
            os.makedirs(outpathTc)
            print outpathTc
        fstr = os.path.join(outpathTc, str(j + find_start))
        print fstr
        try:
            output = subprocess.check_output(['qsub', '-V', '-cwd', '-b y',
                                          exe, fstr, str(Tb), str(Ts), str(Tc)])
            print output
        except:
            print "shit"
        nTc +=1
        
    return n-nq + find_start, nTc

find_start = 480*os.getpid()
numTc = 1.
while True:
    
    find_start, numTc = top_off_que(numinq, find_start, Tc, numTc)
    if numTc>numperT:
        Tc += dTc
        numTc = 1.
    time.sleep(60)
   
