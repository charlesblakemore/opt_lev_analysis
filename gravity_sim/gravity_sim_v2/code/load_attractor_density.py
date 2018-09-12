import numpy as np
import cPickle as pickle
import scipy, sys, time


# Load a regularly-gridded cantilever density data
rhofil = open('./density_grid_1um.txt', 'r')
lines = rhofil.readlines()
rhofil.close()


### CONVERT STUPID COMSOL OUTPUT TO SENSIBLE FORM
# Load grid points
linenum = 0
for line in lines:
	linenum += 1

	if line[0] == '%':
		continue

	if 'xx' not in locals():
		xx = [float(x) for x in line.split()]
		continue
	if 'yy' not in locals():
		yy = [float(x) for x in line.split()]
		continue
	if 'zz' not in locals():
		zz = [float(x) for x in line.split()]
		continue

	if 'xx' in locals() and 'yy' in locals() and 'zz' in locals():
		break

# Extract density from the remainder of the file
rho = np.zeros((len(xx), len(yy), len(zz)), dtype='float')
for j, y in enumerate(yy):
	for k, z in enumerate(zz):
		line = lines[j + k*len(yy) + linenum - 1]
		newline = np.array([float(x) for x in line.split()])
		badpts = np.isnan(newline)
		newline[badpts] = 1.0e-9

		rho[:,j,k] = newline


		# Useful to make sure finding all densities
		if 'vals' not in locals():
			vals = np.unique(rho[:,j,k])
		else:
			vals = np.hstack((vals, np.unique(rho[:,j,k])))


def adjust_rho(rho, xx, yy, zz):
	for j, y in enumerate(yy):
		for k, z in enumerate(zz):
			line = rho[:,j,k]
			first = line[0]

			hit = False
			nHits = 0

			for i, el in enumerate(line):
				if line[i] != first and not hit:
					line[i-1] = line[i]
					hit = True
					continue

				if line[i] != first and hit:
					continue

				if line[i] == first and hit:
					hit = False

			rho[:,j,k] = line

#adjust_rho(rho, xx, yy, zz)



def clean_up_line(line):
	changed = False
	Nchanged = 0
	first = line[0]
	for k, el in enumerate(line):
		if k == 0 or k == (len(line) - 1):
			continue

		if line[k] != first and not changed:
			changed = True
			Nchanged += 1
		elif line[k] != first and changed:
			Nchanged += 1
			if Nchanged >= 5:
				first = line[k]
				changed = False
				Nchanged = 0

		if line[k] == first and changed:
			if Nchanged < 6:
				for ind in range(Nchanged):
					line[k-ind-1] = first
			changed = False
			Nchanged = 0

	line_r = line[::-1]
	changed = False
	Nchanged = 0
	for k, el in enumerate(line_r):
		if k == 0 or k == (len(line_r) - 1):
			continue

		if line_r[k] != first and not changed:
			changed = True
			Nchanged += 1
		elif line_r[k] != first and changed:
			Nchanged += 1
			if Nchanged >= 5:
				first = line_r[k]
				changed = False
				Nchanged = 0

		if line_r[k] == first and changed:
			if Nchanged < 6:
				for ind in range(Nchanged):
					line_r[k-ind-1] = first
			changed = False
			Nchanged = 0

	line = line_r[::-1]
	return line

def clean_up_rho(rho, xx, yy, zz):
	for i, x in enumerate(xx):
		for j, y in enumerate(yy):
			newline = clean_up_line(rho[i,j,:])
			rho[i,j,:] = newline[:]

	for i, x in enumerate(xx):
		for k, z in enumerate(zz):
			newline = clean_up_line(rho[i,:,k])
			rho[i,:,k] = newline[:]

	for j, y in enumerate(yy):
		for k, z in enumerate(zz):
			newline = clean_up_line(rho[:,j,k])
			rho[:,j,k] = newline[:]



print 'Cleaning up density profile...'
sys.stdout.flush()
clean_up_rho(rho, xx, yy, zz)


pickle.dump((rho, xx, yy, zz), open('rho_arr.p', 'wb'))
