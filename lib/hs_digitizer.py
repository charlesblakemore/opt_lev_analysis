import h5py
import numpy as np 
import glob
from scipy.optimize import curve_fit

import bead_util as bu


def copy_dict(dic):
    out_dict = {}
    for k in list(dic.keys()):
        out_dict[k] = dic[k]
    return out_dict


class hsDat:
    def __init__(self, fname='', load=False, load_attribs=True):
        self.fname = fname
        self.dat = []
        self.attribs = {}

        if load:
            self.load()
        if load_attribs:
            self.load_attribs()


    def load(self, fname=''):
        if len(fname):
            self.fname = fname

        try:
            f = h5py.File(self.fname, 'r')
            dset = f['beads/data/high_speed_data']
            self.dat = np.transpose(dset)
            f.close()
        except (KeyError, IOError):
            print("Warning, got no keys for: ", self.fname)
            self.dat = []


    def load_attribs(self, fname=''):
        if len(fname):
            self.fname = fname

        try:
            self.attribs = \
                bu.load_xml_attribs(self.fname, types=['I32', 'DBL', 'Array'])
        except (KeyError, IOError):
            print("Warning, got no attribs for: ", self.fname)
            attribs = {}





