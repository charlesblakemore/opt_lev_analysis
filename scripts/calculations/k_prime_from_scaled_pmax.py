import numpy as np
import damping

save = False
ringdown_path = '/home/dmartin/analyzedData/20190626/ringdown/after_pramp/after_pramp.npz'
mass_path = '/calibrations/masses/20190626_lowp_neg.mass'
pmax_fit_path = '/home/dmartin/analyzedData/20190626/pramp/pmax_vs_mass_fit.npz'

data = np.load(ringdown_path)
mass_arr = np.load(mass_path) 
pmax_fit = np.load(pmax_fit_path)

ms_mass = mass_arr[0]
m = ms_mass
ms_mass_stat_err = mass_arr[1]
ms_mass_sys_err = mass_arr[2]

print('mass',m,ms_mass_stat_err,ms_mass_sys_err)

rho = 1.550e3
rho_err = 0.08 #From Chas' MS mass paper

#Dipole scaled pmax coef
inv_sqrt_coef = pmax_fit['a_coeff']#1.83234e-3 # (V/m)(e um)/(rad/s)(sqrt(amu)Torr)
inv_sqrt_coef_err = pmax_fit['a_coeff_err']
print(inv_sqrt_coef)

V = 4. #Vpp
V_err = 4.05 #V from datasheet. 2% of full amplitude (+/-200V)

E = 4. * 50. * 0.66 * (1/4.e-3) # V/m
E_sys_err = E * (1/(50*V)) * V_err #multiply V by 100 (Tabor monitor factor) and divide by 2 to get peak value

drive_freq = 100. * np.pi *1000. # rad/s

#Tau from ringdown measurement
tau = data['tau'] #s 
tau_stat_err = data['tau_err']

mass = 18.0153 * 1.660e-27 # kg. Assuming nitrogen as main mass content which could be wrong since water is typically main content when chamber is at base

A = (1.602176634e-19) *  (1./np.sqrt(1.660539066e-27)) * (1.e-6) #Conversion from inv_sqrt_coef units to m^4/sqrt(Joules). Did not use torr to Pa conversion here because we want units of torr in the end anyway. Constants from NIST

kprime = E/(drive_freq*inv_sqrt_coef)
kprime *= A # m^4/sqrt(joules)

kprime_stat_err = kprime * (1/inv_sqrt_coef) * (inv_sqrt_coef)**2 
kprime_sys_err = kprime*(1/E)*(E_sys_err)

kprime_theory = damping.rotBeta()

print('kprime',kprime/133.3224, kprime_stat_err/133.3224,kprime_sys_err/133.3224, damping.rotBeta()) #1 Torr = 133.3224 Pa

#Microsphere properties
r = (0.75 * m/(np.pi*rho))**(1./3.) # m
r_stat_err = r * np.sqrt((ms_mass_stat_err/m)**2 + (rho_err/rho)**2)
r_sys_err = r * np.sqrt((ms_mass_stat_err/m)**2) 

print('radius',r,r_stat_err,r_sys_err)
 
I = 0.4 * m * r**2 # kg m^2
I_stat_err = np.sqrt((0.4*r**2*ms_mass_stat_err)**2 + (0.8*m*r*r_stat_err)**2)
I_sys_err = np.sqrt((0.4*r**2*ms_mass_sys_err)**2 + (0.8*m*r*r_sys_err)**2)

print('moment of inertia',I,I_stat_err,I_sys_err,tau_stat_err)


P = I/(tau*np.sqrt(mass)*kprime)
P_stat_err = P * np.sqrt((I_stat_err/I)**2 + (tau_stat_err/tau)**2 + (kprime_stat_err/kprime)**2)
P_sys_err = P * np.sqrt((I_sys_err/I)**2 + (kprime_sys_err/kprime)**2) 


#if save:
#	np.savez(out
#all_calculations = {'kprime':kprime, 'kprime_sys_err':kprime_sys_err, 'kprime_stat_err': kprime_stat_err,kprime_theory':kprime_theory,'Pressure': P,'P_stat_err':P_stat_err,'P_sys_err': P_sys_err}

print('pressure',P,P_stat_err,P_sys_err)
