import sys, os

import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

plt.rcParams.update({'font.size': 14})


base_path = '/data/new_trap_processed/mockfiles/'

meas1 = '20200320/results/noise/Batch3/Akio_Nadav_Chas_MLE_2.p'
meas2 = '20200320/results/noise/Batch3/Akio_Nadav_Chas_Consv2.p'

mle_dat = pickle.load( open(os.path.join(base_path, meas1), 'rb') )
cons_dat = pickle.load( open(os.path.join(base_path, meas2), 'rb') )



base_result_path = '/home/cblakemore/tmp/'

result1 = 'signal_injection_batch3_discovery_unc_3.p'
result2 = 'signal_injection_batch3_conservative_3.p'

mle_result = pickle.load( open(os.path.join(base_result_path, result1), 'rb') )
# cons_result = pickle.load( open(os.path.join(base_result_path, result2), 'rb') )


result3 = 'signal_injection_batch3_no-sig_discovery_3.p'

noise_result = pickle.load( open(os.path.join(base_result_path, result3), 'rb') )
noise_key = list(noise_result.keys())[0]

lambdas = np.array([5.0, 10.0, 12.0, 18.0, 20.0, 25.0, 31.0])
alphas = []
result_alphas = []
result_upper = []
result_lower = []
noise_alphas = []

indices = []

for index, row in mle_dat.iterrows():
    if row['phase_val'] !=  'TF':
        continue
    # print(row['phase_val'])

    indices.append(index)
    # result_index = row['index']
    result_index = row['run_no_true']
    # print(index, result_index)
    lambda_index = np.argmin( np.abs(lambdas - row['lambda']) )
    alpha = row['alpha']

    mle = mle_result[str(result_index)]
    mle_unc  = mle_result[str(result_index)+'_unc']
    result_lower.append(mle_unc[2,0,lambda_index])
    result_upper.append(mle_unc[2,1,lambda_index])

    noise_alpha_both = noise_result[noise_key][2,:,lambda_index]
    max_ind = np.argmax( np.abs(noise_alpha_both) )

    noise_alphas.append( noise_alpha_both[max_ind] )

    if row['direction'] == 'pull':
        alphas.append( alpha )
        result_alphas.append( mle[2,0,lambda_index] )
    if row['direction'] == 'push':
        alphas.append( -1.0*alpha )
        result_alphas.append( mle[2,1,lambda_index] )

alphas = np.array(alphas)
result_alphas = np.array(result_alphas)
result_uncs =  np.abs(np.array([result_lower, result_upper]))
noise_alphas = np.array(noise_alphas)

plt.figure(figsize=(10,5))
plt.scatter(indices, alphas, marker='x', s=75, label='Injected Truth', color='k')
plt.errorbar(indices, result_alphas, yerr=result_uncs, \
             capsize=3, fmt='o', ms=5, label='MLE w/ 90% CL')
plt.scatter(indices, result_alphas - noise_alphas, color='C1', s=25, label='MLE - "noise"')
plt.xlabel('Signal Injection Index [arb]')
plt.ylabel('$\\alpha$ [abs]')
plt.legend()
plt.tight_layout()

plt.show()
