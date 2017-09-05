import numpy as np
import matplotlib.pyplot as pyplot
import cPickle as pickle
import matplotlib.colors as colors
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import sys, time
import cant_utils as cu
import scipy.signal as signal


#######################################################
# This module allows a user to load and interact with
# force curves from a Yukawa-type modification to gravity
# between a microsphere and some probe mass
#
# This assumes the curves have already been computed 
# by code in .../scripts/gravity_sim and simply interacts
# with the pickled result
#######################################################


class Grav_force_curve:
    '''Class to hold the expected force curves for a yukawa modification to gravity.
       Instances of this class take pickled data from ../gravity_sim/save_force_curves.py
       and interpolate over separation and position along cantilever face. Class 
       methods allow one to access and call these interpolating functions in a somewhat
       sensible manner.'''

    def __init__(self, path='', make_splines=False, spline_order=2):

        if( path != ''):
            self.path = path
            self.load_fcurves(path, make_splines=make_splines, \
                              spline_order=spline_order)


    def load_fcurves(self, path='', make_splines=False, spline_order=2):
        if( path == ''):
            print "ERROR: No file to load..."
            return

        try:
            self.path = path
            self.dic = pickle.load( open(path, 'rb') )
            assert self.dic['order'] == 'Rbead, Sep, Yuklambda', 'Key ordering unexpected'
            self.posvec = np.array(self.dic['posvec'])

            rbeads = [rbead for rbead in self.dic.keys() \
                           if isinstance(rbead, (int, float, long))]
            rbeads.sort()
            self.rbeads = np.array(rbeads)
    
            seps = self.dic[self.rbeads[0]].keys()
            seps.sort()
            self.seps = np.array(seps)

            lambdas = self.dic[self.rbeads[0]][self.seps[0]].keys()
            lambdas.sort()
            self.lambdas = np.array(lambdas)

            if make_splines:
                self.make_splines(spline_order=spline_order)

        except:
            self.dic = 'No data loaded!'
            print "ERROR: Bad File! Couldn't load..."

    def make_splines(self, spline_order=2):
        '''Fuction to generate interpolating splines from loaded force curves
               INPUTS: order, polynomial order of interpolating spline

               OUTPUTS: Nothing returned, creates class attribute "splines"'''
        if type(self.dic) == str:
            print "ERROR: No Data Loaded!"
            return

        print "Making splines!"

        Gsplines = {}
        yuksplines = {}

        # Loop over bead radii from loaded data
        for rbead in self.rbeads:
            if rbead not in Gsplines.keys():
                Gsplines[rbead] = {}
                yuksplines[rbead] = {}

            # Loop over length scale lambda from yukawa modification
            for yuklambda in self.lambdas:

                # Finally, loop over separations and construct a grid
                Ggrid = []
                yukgrid = []
                for sep in self.seps:

                    # Find the appropriate curve
                    Gcurve, yukcurve = self.dic[rbead][sep][yuklambda]

                    # Stack the data
                    try:
                        Ggrid = np.vstack((Ggrid, Gcurve))
                        yukgrid = np.vstack((yukgrid, yukcurve))
                    except:
                        Ggrid = Gcurve
                        yukgrid = yukcurve

                # Make an interpolating object which takes advantage of 
                # regular spacing of both the separation and the position
                # along the cantilever
                Ginterpfunc = interpolate.RectBivariateSpline(self.seps, self.posvec, Ggrid, \
                                                              kx=spline_order, ky=spline_order)
                yukinterpfunc = interpolate.RectBivariateSpline(self.seps, self.posvec, yukgrid, \
                                                                kx=spline_order, ky=spline_order)

                Gsplines[rbead][yuklambda] = Ginterpfunc
                yuksplines[rbead][yuklambda] = yukinterpfunc
                
        self.splines = (Gsplines, yuksplines)


    def mod_grav_force(self, xarr, sep=10.0e-6, alpha=1., yuklambda=1.0e-6, rbead=2.43e-6, \
                       verbose=False, nograv=False):
        '''Returns a modified gravity force curve for a given separation,
           alpha and lambda. Includes regular gravity
               INPUTS: xarr [m], array of x points to compute force, x=0 center of cant.
                       sep [m], bead cantilever separation
                       alpha [abs], strength relative to gravity
                       yuklambda [m], length scale of modifications 
                       rbead [m], bead radius
                       
               OUTPUTS: numpy array [N], force along points in xarr.'''

        # make sure rbead and yuklambda exist as keys
        if rbead not in self.rbeads:
            close_ind = np.argmin( np.abs(rbead - self.rbeads) )
            new_rbead = self.rbeads[close_ind]
            rbead = new_rbead
            if verbose:
                print "Couldn't find rbead you wanted... Using rbead = %0.3g" % new_rbead

        if yuklambda not in self.lambdas:
            close_ind = np.argmin( np.abs(yuklambda - self.lambdas) )
            new_lambda = self.lambdas[close_ind]
            yuklambda = new_lambda
            if verbose:
                print "Couldn't find scale you wanted... Using lambda = %0.3g" % new_lambda

        # Identify splines for given rbead and yuklambda
        Gfunc = self.splines[0][rbead][yuklambda]
        yukfunc = self.splines[1][rbead][yuklambda]
        
        # Compute newtownian gravity and yukawa modification with alpha = 1
        Gcurve = Gfunc(sep, xarr)
        yukcurve = yukfunc(sep, xarr)

        # Compute the total force with given alpha
        if nograv:
            totforce = alpha * yukcurve
        else:
            totforce = Gcurve + alpha * yukcurve

        # Reshape the output to match standard 1D numpy array shape
        totforce_shaped = totforce.reshape( (len(xarr),) )
        return totforce_shaped



    def mod_grav_force_point(self, pos, sep=10., alpha=1., yuklambda=1.0e-6, rbead=2.43e-6, \
                             verbose=False, nograv=False):
        '''Returns the force at a particular point for a given separation, alpha,
           lambda and bead radius. Includes regular gravity.
               INPUTS: pos [m], position along cantilever to compute force, x=0 center
                       sep [m], bead cantilever separation
                       alpha [abs], strength relative to gravity
                       yuklambda [m], length scale of modifications 
                       rbead [m], bead radius
                       
               OUTPUTS: force [N], force at specified point.'''

        # make sure rbead and yuklambda exist as keys
        if rbead not in self.rbeads:
            close_ind = np.argmin( np.abs(rbead - self.rbeads) )
            new_rbead = self.rbeads[close_ind]
            rbead = new_rbead
            if verbose:
                print "Couldn't find bead you wanted... Using rbead = %0.3g" % new_rbead

        if yuklambda not in self.lambdas:
            close_ind = np.argmin( np.abs(yuklambda - self.lambdas) )
            new_lambda = self.lambdas[close_ind]
            yuklambda = new_lambda
            if verbose:
                print "Couldn't find scale you wanted... Using lambda = %0.3g" % new_lambda
            
        # Identify splines for given rbead and yuklambda
        Gfunc = self.splines[0][rbead][yuklambda]
        yukfunc = self.splines[1][rbead][yuklambda]
        
        # Compute newtownian gravity and yukawa modification with alpha = 1
        Gforce = Gfunc(sep, pos)
        yukforce = yukfunc(sep, pos)
        
        # Compute the total force with given alpha
        if nograv:
            totforce = alpha * yukforce
        else:
            totforce = Gforce + alpha * yukforce


        return totforce


    
        



