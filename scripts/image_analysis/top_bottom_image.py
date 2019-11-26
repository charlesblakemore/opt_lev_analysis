import numpy as np
import matplotlib.pyplot as plt
from image_util import *

iref = "/data/20180622/imaging_tests/top_laser_stage-X80um-Y40um-Z25um.h5.npy"
itrans = "/data/20180622/imaging_tests/bottom_laser_stage-X80um-Y40um-Z25um.h5.npy"

Imgref = Image(iref)
Imgtrans = Image(itrans)

