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

index_start = 0
index_end = z_len-1

for index in range(0, z_len):
    if np.count_nonzero(~np.isnan(mri_data[:, :, index])) == 0:
        index_start = index
    else:
        break

for index in range(z_len-1, -1, -1):
    if np.count_nonzero(~np.isnan(mri_data[:, :, index])) == 0:
        index_end = index
    else:
        break

print("Only considering relevant slices between indices: [" + str(index_start) + "-" + str(index_end) + "]")
mri_data = mri_data[:, :, index_start:index_end+1]
[x_len, y_len, z_len] = mri_data.shape
whole_volume = x_len * y_len * z_len



mri_mask = np.nan_to_num(mri_data)
mri_mask[mri_mask > 0] = 1



#code.interact(local=dict(globals(), **locals()))
#raise Exception()


print("_____________")

loop_num = 16
loop_len = 512

for il in range(0, loop_num):
    ''' Debug purposed printing '''
    print('.', end='')
    if np.remainder(il+1, 32) == 0:
        print(' ' + str(il+1) + '/' + str(loop_num)) # Print newline


    print(il*loop_len)
    print((il*loop_len)+loop_len)

    ''' Only process sub-array '''
    source_patches_loop = volume_source_patch_cuda_all[il*loop_len:(il*loop_len)+loop_len,:]

    '''  SUBTRACTION '''
    sub_result_gm = cuda.device_array((source_patches_loop.shape[0],
                                        target_patches_np_cuda_all.shape[0],
                                        target_patches_np_cuda_all.shape[1]))
    TPB = (4,256)
    BPGx = int(math.ceil(source_patches_loop.shape[0] / TPB[0]))
    BPGy = int(math.ceil(target_patches_np_cuda_all.shape[0] / TPB[1]))
    BPGxy = (BPGx,BPGy)
    cu_sub_st[BPGxy,TPB](source_patches_loop, target_patches_np_cuda_all, sub_result_gm)

    '''  MAX-MEAN-ABS '''
    sub_max_mean_result = cuda.device_array((source_patches_loop.shape[0],
                                                target_patches_np_cuda_all.shape[0],2))
    cu_max_mean_abs[BPGxy,TPB](sub_result_gm, sub_max_mean_result)
    sub_result_gm = 0  # Free memory

    '''  DISTANCE '''
    distances_result = cuda.device_array((source_patches_loop.shape[0],
                                            target_patches_np_cuda_all.shape[0]))
    cu_distances[BPGxy,TPB](sub_max_mean_result,
                            icv_source_flag_valid[il*loop_len:(il*loop_len)+loop_len],
                            distances_result, alpha)
    sub_max_mean_result = 0  # Free memory

    ''' SORT '''
    TPB = 256
    BPG = int(math.ceil(distances_result.shape[0] / TPB))
    cu_sort_distance[BPG,TPB](distances_result)

    ''' MEAN (AGE-VALUE) '''
    idx_start = 8 # Starting index of mean calculation (to avoid bad example)
    distances_result_for_age = distances_result[:,idx_start:idx_start+num_mean_samples]
    distances_result = 0  # Free memory
    cu_age_value[BPG,TPB](distances_result_for_age,
                            age_values_valid[il*loop_len:(il*loop_len)+loop_len])
    distances_result_for_age = 0  # Free memory
    del source_patches_loop  # Free memory
print(' - Finished!\n')


code.interact(local=dict(globals(), **locals()))
