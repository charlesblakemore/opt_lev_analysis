#!/usr/bin/python

import numpy as np
from vector import *

def rk4(xi_old, t, delt, system):
    '''Implements one step of the 4th Order Runge-Kutta method for numerically
    solving a system of first order ODE's'''
    k1 = delt * system(xi_old, t)
    k2 = delt * system(xi_old + k1 / 2, t + (delt / 2))
    k3 = delt * system(xi_old + k2 / 2, t + (delt / 2))
    k4 = delt * system(xi_old + k3, t + delt)

    xi_new = xi_old + (1. / 6.) * (k1 + 2*k2 + 2*k3 + k4)

    return xi_new

def exp(xi_old, t, delt, system):
    '''Implements one step of the explicit Euler method for solving
    a system of ODEs.'''
    xi_new = xi_old + delt * system(xi_old, t)
    return xi_new

def mp(xi_old, t, delt, system):
    '''Implements one step of the midpoint method for solving a system
    of ODEs.'''
    # Guess the midpoint of our step
    xi_tilde = xi_old + (delt * 0.5) * system(xi_old, t)

    # Using this midpoint guess, guess the value of xi_new, our system
    # after this single step of the midpoint method
    xi_new = xi_old + delt * system(xi_tilde, t)
    return xi_new

def stepper(xi_0, ti, tf, delt, system, method):
    '''Repeatedly calls method to solve an ODE from ti to tf, where system
    is a fucntion that returns a vector with our first order derivatives.'''
    # Create a discrete-time array from ti to tf with spacing delt
    tt = np.arange(ti, tf + delt, delt)

    # Initialize list of 'points' which will contain the solution to our
    # ODE at each time in our discrete-time array
    points = []
    i = 0
    xi_old = xi_0
    for t in tt:
        # Using the 4th-order Runge-Kutta method, we evaluate the solution
        # iteratively for each time t in our discrete-time array
        xi_new = method(xi_old, t, delt, system)
        xi_old = xi_new
        points.append(xi_new)
        if (t / tf) > (i * 0.1):
            print(i)
            i += 1

    return tt, points
