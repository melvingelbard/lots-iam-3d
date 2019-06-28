import matplotlib
matplotlib.use('Agg')

from numba import cuda
from timeit import default_timer as timer
from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.morphology as skimorph
import skimage.filters as skifilters
import scipy.ndimage.morphology as scimorph

import math, numba, cv2, csv, gc
import os, errno, sys, shutil

import code, time


print("COUCOUCO")


mri_data = sio.loadmat("D:/Edinburgh/dissertation/V1//flair.mat")            # Loading FLAIR
mri_data = mri_data["flair"]
[x_len, y_len, z_len] = mri_data.shape

print(x_len)

print("_____________")
