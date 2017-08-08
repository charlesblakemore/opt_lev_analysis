import numpy as np
import matplotlib.pyplot as pyplot
import cPickle as pickle
import matplotlib.colors as colors
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import sys, time


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

    def __init__(self, fcurve_path):
        self.path = fcurve_path
        self.dic = pickle.load( open(fcurve_path, 'rb') )
        assert self.dic['order'] == 'Rbead, Sep, Yuklambda', 'Key ordering unexpected'
        self.posvec = self.dic['posvec']
        self.rbeads = [rbead for rbead in self.dic.keys() if isinstance(rbead, (int, float, long))]
        self.rbeads.sort()
        self.seps = self.dic[self.rbeads[0]].keys()
        self.seps.sort()
        self.lambdas = self.dic[self.rbeads[0]][self.seps[0]].keys()
        self.lambdas.sort()

    def make_splines(self):
        Gsplines = {}
        yuksplines = {}
        for rbead in self.rbeads:
            if rbead not in gsplines.keys():
                Gsplines[rbead] = {}
                yuksplines[rbead] = {}
            for yuklambda in self.lambdas:

                for sep in self.seps:
                    Gcurve, yukcurve = self.dic[rbead][sep][yuklambda]
                    try:
                        Ggrid = np.vstack((Ggrid, Gcurve))
                        yukgrid = np.vstack((yukgrid, yukcurve))
                    except:
                        Ggrid = Gcurve
                        yukgrid = yukcurve

                Ginterpfunc = interpolate.RectBivariateSpline(self.posvec, self.seps, Ggrid)
                yukinterpfunc = interpolate.RectBivariateSpline(self.posvec, self.seps, yukgrid)

                Gsplines[rbead][yuklambda] = Ginterpfunc
                yuksplines[rbead][yuklambda] = yukinterpfunc
                
        self.splines = (Gsplines, yuksplines)


    
        
