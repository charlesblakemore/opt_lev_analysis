import h5py
import numpy as np 
import glob
from scipy.optimize import curve_fit

import bead_util as bu #import xmltodict


def copy_dict(dic):
    out_dict = {}
    for k in dic.keys():
        out_dict[k] = dic[k]
    return out_dict


class hsDat:
    def __init__(self, fname):
        try: 
            f = h5py.File(fname,'r')
            dset = f['beads/data/high_speed_data']
            self.dat = np.transpose(dset)
            self.attribs = copy_dict(dset.attrs)
            f.close()

            if not len(self.attribs):
                self.attribs = \
                    bu.load_xml_attribs(fname, types=['I32', 'DBL', 'Array'])

        except (KeyError, IOError):
            print "Warning, got no keys for: ", fname
            dat = []
            attribs = {}
            f = []

