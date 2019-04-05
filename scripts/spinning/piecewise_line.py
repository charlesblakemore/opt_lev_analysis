import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def pw_line(x, b0, b1, b2, y0, m0, m1, m2, m3):
    out_arr = np.zeros(len(x))
    out_arr[x<b0] =  m0*x[x<b0] + y0
    out_arr[(x>=b0)*(x<b1)] = m0*b0 + y0 + m1*(x[(x>=b0)*(x<b1)]-b0)
    out_arr[(x>=b1)*(x<b2)] = m0*b0 + y0 + m1*(b1-b0) + \
                              m2*(x[(x>=b1)*(x<b2)]-b1)
    out_arr[x>=b2] = m0*b0 + y0 + m1*(b1-b0) + m2*(b2-b1) + m3*(x[x>=b2]-b2)
    return out_arr





