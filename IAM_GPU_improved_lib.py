import matplotlib
matplotlib.use('Agg')

from numba import cuda
from timeit import default_timer as timer
from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IAM_lib import *

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.morphology as skimorph
import skimage.filters as skifilters
import scipy.ndimage.morphology as scimorph

import math, numba, cv2, csv, gc
import os, errno, sys, shutil

import code
from PIL import Image

# Turn interactive plotting off
plt.ioff()

def iam_lots_gpu_compute(csv_filename="", patch_size=[1,2,4,8],
                         blending_weights=[0.65,0.2,0.1,0.05], num_sample=[512],
                         alpha=0.5, thrsh_patches = True, bin_tresh=0.5, save_jpeg=True,
                         delete_intermediary=False, nawm_preprocessing=False):
    '''
    FUNCTION'S SUMMARY:

    Main function of the LOTS-IAM-GPU algorithm. This function produces (i.e. saving)
    age maps that indicate level of irregularity of voxels in brain FLAIR MRI. This
    function reads a list of FLAIR MR image (NifTI), ICV mask (NifTI), CSF mask (NifTI),
    NAWM mask (NifTI), and Cortical mask (NifTI) to produce the corresponding age maps
    from a CSV file. Please note that this version only accept NifTI (.nii/.nii.gz) files.

    NOTE: NAWM and Cortical masks are optional. They will be used if they are included
    in the CSV file.

    Format of the CSV input file (NOTE: spaces are used to make the format clearer):

        path_to_mri_codebase_folder, mri_code_name, path_FLAIR, path_ICV, path_CSF,
        path_NAWM (optional), path_Cortical (optional)

    Example (NOTE: spaces are used to make the format clearer):

        /dir/MRI_DB/, MRI001, /dir/MRI_DB/MRI001/FLAIR.nii.gz, /dir/MRI_DB/MRI001/ICV.nii.gz,
        /dir/MRI_DB/MRI001/CSF.nii.gz, /dir/MRI_DB/MRI001/NAWM.nii.gz (optional),
        /dir/MRI_DB/MRI001/Cortex.nii.gz (optional)

    By default, the age maps are calculated by using four different sizes of source/target
    patches (i.e. 1x1, 2x2, 4x4, and 8x8) and 64 target samples. Furthermore, all intermediary
    files are saved in .mat (Matlab) and JPEG files.


    INPUT PARAMETERS:

    This function's behavior can be set by using input parameters below.

        1. output_filedir   : Path of directory for saving all results. Format of the path:
                              "output_path/name_of_experiment"

        2. csv_filename     : Name of a CSV input file which contains list all files to be
                              processed by the LOTS-IAM-GPU. Example: "input.csv"

        3. patch_size       : Size of source/target patches for IAM's computation. Default:
                              [1,2,4,8] to calculate age maps from four different sizes of
                              source/target patches i.e. 1x1, 2x2, 4x4, and 8x8. The sizes
                              of source/target patches must be in the form of python's list.

        4. blending_weights : Weights used for blending age maps produced by different size of
                              source/target patches. The weights must be in the form of python's
                              list, summed to 1, and its length must be the same as `patch_size`.

        5. num_sample       : A list of numbers used for randomly sampling target patches to be
                              used in the LOTS-IAM-GPU calculation. Default: [512]. Available
                              values: [64, 128, 256, 512, 1024, 2048]. Some important notes:

                                a. Smaller number will make computation faster.
                                b. Input the numbers as a list to automatically produce
                                   age maps by using all different numbers of target patches.
                                   The software will automatically create different output
                                   folders for different number of target samples.
                                c. For this version, only 64, 128, 256, 512, 1024, and 2048
                                   can be used as input numbers (error will be raised if other
                                   numbers are used).

        6. alpha            : Weight of distance function to blend maximum difference and
                              average difference between source and target patches. Default:
                              0.5. Input value should be between 0 and 1 (i.e. floating points).
                              The current distance function being used is:

                                  d = (alpha . |max(s - t)|) + ((1 - alpha) . |mean(s - t)|)

                              where d is distance value, s is source patch, and t is target patch.

        7. bin_tresh        : Threshold value for cutting of probability values of brain masks,
                              if probability masks are given instead of binary masks.

        8. save_jpeg        : True  --> Save all JPEG files for visualisation.
                              False --> Do not save the JPEG files.

        9. delete_intermediary : False --> Save all intermediary files (i.e. JPEG/.mat files).
                                 True  --> Delete all intermediary files, saving some spaces in
                                           the hard disk drive.

    OUTPUT:

    The software will automatically create a new folder provided in "output_filedir" variable.
    Please make sure that the directory is accessible and writable.

    Inside the experiment’s folder, each patient/MRI mri_code will have its own folder. In default,
    there are 6 sub-folders which are:
    1. 1: Contains age maps of each slice generated by using 1x1 patch.
    2. 2: Contains age maps of each slice generated by using 2x2 patch.
    3. 4: Contains age maps of each slice generated by using 4x4 patch.
    4. 8: Contains age maps of each slice generated by using 8x8 patch.
    5. IAM_combined_python: Contains two sub-folders:
        a. Patch: contains visualisation of age maps of each slices in JPEG files, and
        b. Combined: contains visualisation of the final output of LOTS-IAM-GPU’s computation.
    6. IAM_GPU_nifti_python: Contains one Matlab (.mat) file and three NIfTI files (.nii.gz):
        a. all_slice_dat.mat: processed mri_code of all slices in Matlab file,
        b. IAM_GPU_COMBINED.nii.gz: the original age map values,
        c. IAM_GPU_GN.nii.gz: the final age map values (i.e. GN and penalty), and
        d. IAM_GPU_GN_postprocessed.nii.gz: the final age map values plus post-processing
           (only if NAWM mask is provided).

    Note: If parameter value of `delete_intermediary` is `True`, then all folders listed above
    will be deleted, except for folder `IAM_GPU_nifti_python` and its contents.

    MORE HELP:

    Please read README.md file provided in:
        https://github.com/febrianrachmadi/lots-iam-gpu

    VERSION (dd/mm/yyyy):
    - 31/05/2018b: NAWM and Cortical brain masks are now optional input (will be used if available).
    - 31/05/2018a: Fix header information of the LOTS-IAM-GPU's result.
    - 08/05/2018 : Add lines to cutting off probability mask and deleting intermediary folders.
    - 07/05/2018 : Initial release code.
    '''

    ## Check availability of input files and output path
    if csv_filename == "":
        raise ValueError("Please set output folder's name and CSV mri_code filename. See: help(iam_lots_gpu)")
        return 0

    ## Check compatibility between 'patch_size' and 'blending_weights'
    if len(patch_size) != len(blending_weights):
        raise ValueError("Lengths of 'patch_size' and 'blending_weights' variables are not the same. Length of 'patch_size' is " + str(len(patch_size)) + ", while 'blending_weights' is " + str(len(blending_weights)) + ".")
        return 0

    ## If intermediary files to be deleted, don't even try to save JPEGs
    if delete_intermediary:
        save_jpeg = False

    ''' Set number of mean samples automatically '''
    ''' num_samples_all = [64, 128, 256, 512, 1024, 2048] '''
    ''' num_mean_samples_all = [16, 32, 32, 64, 128, 128] '''
    num_samples_all = num_sample
    num_mean_samples_all = []
    for sample in num_samples_all:
        if sample == 64:
            num_mean_samples_all.append(16)
        elif sample == 128:
            num_mean_samples_all.append(32)
        elif sample == 256:
            num_mean_samples_all.append(32)
        elif sample == 512:
            num_mean_samples_all.append(64)
        elif sample == 1024:
            num_mean_samples_all.append(128)
        elif sample == 2048:
            num_mean_samples_all.append(128)
        else:
            raise ValueError("Number of samples must be either 64, 128, 256, 512, 1024 or 2048!")
            return 0

    print("--- PARAMETERS - CHECKED ---")
    print('CSV mri_code filename: ' + csv_filename)
    print('Patch size(s): ' + str(patch_size))
    print('Number of samples (all): ' + str(num_samples_all))
    print('Number of mean samples (all): ' + str(num_mean_samples_all))
    print('Save JPEGs? ' + str(save_jpeg))
    print("--- PARAMETERS - CHECKED ---\n")

    for ii_s in range(0, len(num_samples_all)):
        num_samples = num_samples_all[ii_s]
        num_mean_samples = num_mean_samples_all[ii_s]
        print('Number of samples for IAM: ' + str(num_samples))
        print('Number of mean samples for IAM: ' + str(num_mean_samples))

        with open(csv_filename, newline='') as csv_file:
            num_subjects = len(csv_file.readlines())
            print('Number of subject(s): ' + str(num_subjects))

        with open(csv_filename, newline='', encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file)

            timer_idx = 0
            elapsed_times_all = np.zeros((num_subjects))
            elapsed_times_patch_all = np.zeros((num_subjects, len(patch_size)))
            for row in reader:
                mri_code = row[2]

                dirOutput = row[1]
                print('Output dir: ' + dirOutput + '\n--')

                try:
                    os.makedirs(dirOutput)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise


                print('--\nNow processing mri_code: ' + mri_code)

                inputSubjectDir = row[0]
                print('Input filename (full path): ' + inputSubjectDir)

                ''' Create output folder(s) '''
                dirOutData = dirOutput + '/' + mri_code
                dirOutDataCom = dirOutput + '/' + mri_code + '/IAM_combined_python/'
                dirOutDataPatch = dirOutput + '/' + mri_code + '/IAM_combined_python/Patch/'
                dirOutDataCombined = dirOutput + '/' + mri_code + '/IAM_combined_python/Combined/'
                try:
                    print(dirOutDataCom)
                    os.makedirs(dirOutDataCom)
                    os.makedirs(dirOutDataPatch)
                    os.makedirs(dirOutDataCombined)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                mri_data = sio.loadmat(row[0])     # Loading FLAIR
                mri_data = mri_data["flair"]
                [x_len, y_len, z_len] = mri_data.shape

                one_mri_data = timer()
                for xy in range(0, len(patch_size)):
                    print('>>> Processing patch-size: ' + str(patch_size[xy]) + ' <<<\n')

                    try:
                        os.makedirs(dirOutData + '/' + str(patch_size[xy]))
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise

                    one_patch = timer()
                    for zz in range(0, mri_data.shape[2]):
                        print('---> Slice number: ' + str(zz) + ' <---')

                        '''
                        KEY POINT: PRE-PROCESSING P.2 - START
                        -------------------------------------
                        This version still does per slice operation for extracting brain tissues.
                        Two important variables used in the next part of the code are:
                        1. mask_slice --->  Combination of ICV & CSF masks. It is used to find valid source patches
                                            for LOTS-IAM-GPU computation (i.e. brain tissues' source patches).
                        2. brain_slice -->  Brain tissues' information from FLAIR slice.
                        '''

                        mask_slice = np.nan_to_num(mri_data[:, :, zz])
                        mask_slice[mask_slice > 0] = 1

                        brain_slice = np.nan_to_num(mri_data[:, :, zz])

                        '''
                        -----------------------------------
                        KEY POINT: PRE-PROCESSING P.2 - END
                        '''

                        ## Show brain slice to be used for computation
                        #fig, ax = plt.subplots()
                        #cax = ax.imshow(icv_slice, cmap="jet")
                        #cbar = fig.colorbar(cax)
                        #fig.show()
                        #plt.savefig("plot.jpg")


                        # Vol distance threshold
                        vol_slice = np.count_nonzero(brain_slice) / (x_len * y_len)                         ## Proportion of brain slice compared to full image
                        print('DEBUG-Patch: brain_slice - ' + str(np.count_nonzero(brain_slice)) +
                              ', x_len * y_len - ' + str(x_len * y_len) + ', vol: ' + str(vol_slice))       ## x_len/y_len = 512 here

                        # Patch's sampling number treshold
                        TRSH = 0.50
                        if patch_size[xy] == 1:
                            if vol_slice < 0.010: TRSH = 0
                            elif vol_slice < 0.035: TRSH = 0.15
                            elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
                            elif vol_slice >= 0.070: TRSH = 0.80
                        elif patch_size[xy] == 2:
                            if vol_slice < 0.010: TRSH = 0
                            elif vol_slice < 0.035: TRSH = 0.15
                            elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
                            elif vol_slice >= 0.070: TRSH = 0.80
                        elif patch_size[xy] == 4 or patch_size[xy] == 8:
                            if vol_slice < 0.035: TRSH = 0

                        print('DEBUG-Patch: Size - ' + str(patch_size[xy]) + ', slice - ' + str(zz) +
                              ', vol: ' + str(vol_slice) + ', TRSH: ' + str(TRSH))

                        counter_y = int(y_len / patch_size[xy])                                             ## counter_y = 512 if patch of size 1 and image of size 512x512
                        counter_x = int(x_len / patch_size[xy])
                        source_patch_len = counter_x * counter_y                                            ## How many source patches are neede (e.g. for 1, we need one for each pixel)
                        age_values_all = np.zeros(source_patch_len)                                         ## Age Map that will be filled with the actual values

                        valid = 0
                        if ((vol_slice >= 0.008 and vol_slice < 0.035) and (patch_size[xy] == 1 or patch_size[xy] == 2)) or \
                            ((vol_slice >= 0.035 and vol_slice < 0.065) and (patch_size[xy] == 1 or patch_size[xy] == 2 or \
                             patch_size[xy] == 4)) or (vol_slice > 0.065):
                            valid = 1

                            ## Creating grid-patch 'xy-by-xy'
                            #  -- Column
                            y_c = np.ceil(patch_size[xy] / 2)
                            y_c_sources = np.zeros(int(y_len / patch_size[xy]))
                            for iy in range(0, int(y_len / patch_size[xy])):
                                y_c_sources[iy] = (iy * patch_size[xy]) + y_c - 1


                            #  -- Row
                            x_c = np.ceil(patch_size[xy] / 2)
                            x_c_sources = np.zeros(int(x_len / patch_size[xy]))
                            for ix in range(0, int(x_len / patch_size[xy])):
                                x_c_sources[ix] = (ix * patch_size[xy]) + x_c - 1


                            ''' Extracting Source Patches '''
                            area_source_patch = np.zeros([1,patch_size[xy],patch_size[xy]])
                            center_source_patch = np.zeros([1,2])
                            icv_source_flag = np.zeros([source_patch_len])
                            icv_source_flag_valid = np.ones([source_patch_len])
                            index_mapping = np.ones([source_patch_len]) * -1


                            flag = 1
                            index = 0
                            index_source= 0

                            if patch_size[xy] == 1:
                                area_source_patch = brain_slice[mask_slice == 1]
                                area_source_patch = area_source_patch.reshape([area_source_patch.shape[0], 1, 1])
                                index = source_patch_len
                                index_source = area_source_patch.shape[0]
                                icv_source_flag = mask_slice.flatten()
                                positive_indices = (np.where(brain_slice.flatten() > 0))[0]
                                index = 0
                                for i in positive_indices:
                                    index_mapping[i] = index
                                    index += 1

                            else:
                                area_source_patch = []
                                for isc in range(0, counter_x):
                                    for jsc in range(0, counter_y):
                                            icv_source_flag[index] = mask_slice[int(x_c_sources[isc]), int(y_c_sources[jsc])]
                                            if icv_source_flag[index] == 1:
                                                temp = get_area(x_c_sources[isc], y_c_sources[jsc],
                                                                patch_size[xy], patch_size[xy], brain_slice)
                                                area_source_patch.append(temp.tolist())
                                                index_mapping[index] = index_source
                                                index_source += 1

                                            index += 1
                                area_source_patch = np.asarray(area_source_patch)




                            icv_source_flag_valid = icv_source_flag_valid[0:index_source]
                            age_values_valid = np.zeros(index_source)

                            """ TO DELETE, IT'S JUST FOR DISSERTATION
                            for i in range(area_source_patch.shape[2]):
                                plt.imshow(area_source_patch[i]) #Needs to be in row,col order
                                plt.savefig("test.jpg")
                            """


                            ''' Extracting Target Patches '''
                            target_patches = []
                            index_debug = 0
                            random_array = np.random.randint(10, size=(x_len, y_len))
                            index_possible = np.zeros(brain_slice.shape)
                            index_possible[(mask_slice != 0) & (random_array > TRSH*10)] = 1
                            index_possible = np.argwhere(index_possible)


                            for index_chosen in index_possible:
                                x, y = index_chosen
                                area = get_area(x, y, patch_size[xy], patch_size[xy], brain_slice)
                                if area.size == patch_size[xy] * patch_size[xy]:
                                    if np.random.randint(low=1, high=10)/10 < (100/(x*y)) * num_samples:
                                        pass
                                    target_patches.append(area)
                                    index_debug += 1


                            target_patches_np = get_shuffled_patches(target_patches, num_samples)
                            target_patches_np = target_patches_np[0:num_samples,:,:]
                            print('Sampling finished: ' + ' with: ' + str(index_debug) + ' samples from: ' + str(x_len * y_len))
                            area = []

                            ''''''
                            ''' Reshaping array mri_code '''
                            area_source_patch_cuda_all = np.reshape(area_source_patch,(area_source_patch.shape[0],
                                                            area_source_patch.shape[1] * area_source_patch.shape[2]))
                            target_patches_np_cuda_all = np.reshape(target_patches_np, (target_patches_np.shape[0],
                                                            target_patches_np.shape[1] * target_patches_np.shape[2]))

                            #if patch_size[xy] == 2:
                            #    code.interact(local=dict(globals(), **locals()))

                            melvin = timer()
                            source_len = icv_source_flag_valid.shape[0]
                            loop_len = 512 # def: 512
                            loop_num = int(np.ceil(source_len / loop_len))
                            print('\nLoop Information:')
                            print('Total number of source patches: ' + str(source_len))
                            print('Number of voxels processed in one loop: ' + str(loop_len))
                            print('Number of loop needed: ' + str(loop_num))
                            print('Check GPU memory: ' + str(cuda.current_context().get_memory_info()))

                            for il in range(0, loop_num):
                                ''' Debug purposed printing '''
                                print('.', end='')
                                if np.remainder(il+1, 32) == 0:
                                    print(' ' + str(il+1) + '/' + str(loop_num)) # Print newline

                                ''' Only process sub-array '''
                                source_patches_loop = area_source_patch_cuda_all[il*loop_len:(il*loop_len)+loop_len,:]

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
                                #code.interact(local=dict(globals(), **locals()))
                            print(' - Finished!\n')
                            print(timer() - melvin)
                            raise Exception()
                        ''' Mapping from age_value_valid to age value_all '''
                        if valid == 1:
                            index = 0
                            for idx_val in index_mapping:
                                if idx_val != -1:
                                    age_values_all[index] = age_values_valid[int(idx_val)]
                                index += 1

                        ''' Normalisation to probabilistic map (0...1) '''
                        if (np.max(age_values_all) - np.min(age_values_all)) == 0:
                            all_mean_distance_normed = age_values_all
                        else:
                            all_mean_distance_normed = np.divide((age_values_all - np.min(age_values_all)),
                                (np.max(age_values_all) - np.min(age_values_all)))

                        ''' SAVE Result (JPG) '''
                        slice_age_map = np.zeros([counter_x,counter_y])
                        index = 0
                        for ix in range(0, counter_x):
                            for iy in range(0, counter_y):
                                slice_age_map[ix,iy] = all_mean_distance_normed[index]
                                index += 1

                        ## Save mri_data
                        sio.savemat(dirOutData + '/' + str(patch_size[xy]) + '/' + str(zz) + '_dat.mat',
                                    {'slice_age_map':slice_age_map})

                        print('Check GPU memory: ' + str(cuda.current_context().get_memory_info()))
                        print('GPU flushing..\n--\n')
                        numba.cuda.profile_stop()
                    elapsed_times_patch_all[timer_idx,xy] = timer() - one_patch
                    print('IAM for MRI ID: ' + mri_code + ' with patch size: ' + str(patch_size[xy])
                          + ' elapsed for: ' + str(elapsed_times_patch_all[timer_idx,xy]))

                elapsed_times_all[timer_idx] = timer() - one_mri_data
                print('IAM for MRI ID: ' + mri_code + ' elapsed for: ' + str(elapsed_times_all[timer_idx]))
                timer_idx += 1

                ''' Save all elapsed times '''
                sio.savemat(dirOutput + '/elapsed_times_all_' + str(num_samples) + 's' + str(num_mean_samples) + 'm.mat',
                            {'elapsed_times_all':elapsed_times_all})
                sio.savemat(dirOutput + '/elapsed_times_patch_all_' + str(num_samples) + 's' + str(num_mean_samples) + 'm.mat',
                            {'elapsed_times_patch_all':elapsed_times_patch_all})
                ''' IAM's (GPU Part) Computation ENDS here '''

                '''
                KEY POINT: IAM's Combination, Penalisation, and Post-processing - START
                -----------------------------------------------------------------------
                Part 0 - Saving output results in .mat and JPEG files.
                Part 1 - Combination of multiple age maps.
                Part 2 - Global normalisation and penalisation of age maps based on brain tissues.
                Part 3 - Post-processing.

                Hint: You can search the keys of Part 0/1/2/3.
                '''
                combined_age_map_mri = np.zeros((x_len, y_len, z_len))
                combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))
                combined_age_map_mri_mult_normed = np.zeros((x_len, y_len, z_len))
                for zz in range(0, mri_data.shape[2]):
                    mri_slice = mri_data[:,:,zz]
                    mask_slice = np.nan_to_num(mri_slice)
                    mask_slice[mask_slice > 0] = 1
                    penalty_slice = np.nan_to_num(mri_slice)   ### PENALTY

                    slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))

                    dirOutData = dirOutput + '/' + mri_code
                    for xy in range(0, len(patch_size)):
                        mat_contents = sio.loadmat(dirOutData + '/' + str(patch_size[xy]) + '/' + str(zz) + '_dat.mat')
                        slice_age_map = mat_contents['slice_age_map']
                        slice_age_map_res = cv2.resize(slice_age_map, None, fx=patch_size[xy],
                                                       fy=patch_size[xy], interpolation=cv2.INTER_CUBIC)
                        slice_age_map_res = skifilters.gaussian(slice_age_map_res,sigma=0.5,truncate=2.0)
                        #if zz== 20:
                        #    code.interact(local=dict(globals(), **locals()))
                        slice_age_map_res = np.multiply(mask_slice, slice_age_map_res)
                        slice_age_map_all[xy,:,:] = slice_age_map_res
                    slice_age_map_all = np.nan_to_num(slice_age_map_all)


                    if save_jpeg:
                        ''' >>> Part 0 <<<'''
                        ''' Show all age maps based on patch's size and saving the mri_data '''
                        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
                        fig.set_size_inches(10, 10)
                        fig.suptitle('All Patches Gaussian Filtered', fontsize=16)

                        axes[0,0].set_title('Patch 1 x 1')
                        im1 = axes[0,0].imshow(np.rot90(slice_age_map_all[0,:,:]), cmap="jet", vmin=0, vmax=1)
                        divider1 = make_axes_locatable(axes[0,0])
                        cax1 = divider1.append_axes("right", size="7%", pad=0.05)
                        cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

                        if len(patch_size) > 1:
                            axes[0,1].set_title('Patch 2 x 2')
                            im2 = axes[0,1].imshow(np.rot90(slice_age_map_all[1,:,:]), cmap="jet", vmin=0, vmax=1)
                            divider2 = make_axes_locatable(axes[0,1])
                            cax2 = divider2.append_axes("right", size="7%", pad=0.05)
                            cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

                            if len(patch_size) > 2:
                                axes[1,0].set_title('Patch 4 x 4')
                                im3 = axes[1,0].imshow(np.rot90(slice_age_map_all[2,:,:]), cmap="jet", vmin=0, vmax=1)
                                divider3 = make_axes_locatable(axes[1,0])
                                cax3 = divider3.append_axes("right", size="7%", pad=0.05)
                                cbar3 = plt.colorbar(im3, ticks=[0, 0.5, 1], cax=cax3)

                                if len(patch_size) > 3:
                                    axes[1,1].set_title('Patch 8 x 8')
                                    im4 = axes[1,1].imshow(np.rot90(slice_age_map_all[3,:,:]), cmap="jet", vmin=0, vmax=1)
                                    divider4 = make_axes_locatable(axes[1,1])
                                    cax4 = divider4.append_axes("right", size="7%", pad=0.05)
                                    cbar4 = plt.colorbar(im4, ticks=[0, 0.5, 1], cax=cax4)

                        plt.tight_layout()
                        plt.subplots_adjust(top=0.95)

                        ''' >>> Part 0 <<<'''
                        ''' Save mri_data in *_all.jpg '''
                        dirOutData = dirOutput + '/' + mri_code + '/IAM_combined_python/Patch/'
                        fig.savefig(dirOutData + str(zz) + '_all.jpg', dpi=100)
                        print('Saving files: ' + dirOutData + str(zz) + '_all.jpg')
                        plt.close()

                    ''' >>> Part 1 <<< '''
                    ''' Combined all patches age map information '''
                    combined_age_map = 0
                    for bi in range(len(patch_size)):
                        combined_age_map += np.multiply(blending_weights[bi],slice_age_map_all[bi,:,:])
                    combined_age_map_mri[:,:,zz] = combined_age_map

                    ''' Global Normalisation - saving needed mri_data '''
                    combined_age_map_mri_mult[:,:,zz] = np.multiply(np.multiply(combined_age_map, penalty_slice), mask_slice)  ### PENALTY
                    normed_only = np.divide((combined_age_map_mri[:,:,zz] - np.min(combined_age_map_mri[:,:,zz])),\
                                            (np.max(combined_age_map_mri[:,:,zz]) - np.min(combined_age_map_mri[:,:,zz])))
                    normed_mult = np.multiply(np.multiply(normed_only, penalty_slice), mask_slice)  ### PENALTY
                    normed_mult_normed = np.divide((normed_mult - np.min(normed_mult)), \
                                            (np.max(normed_mult) - np.min(normed_mult)))
                    combined_age_map_mri_mult_normed[:,:,zz] = normed_mult_normed

                    ''' Save mri_data in *.mat '''
                    dirOutData = dirOutput + '/' + mri_code + '/IAM_combined_python/Patch/'
                    print('Saving files: ' + dirOutData + 'c' + str(zz) + '_combined.mat\n')
                    sio.savemat(dirOutData + 'c' + str(zz) + '_combined.mat', {'slice_age_map_all':slice_age_map_all,
                                                                'combined_age_map':normed_only,
                                                                'mri_slice_mul_normed':normed_mult_normed,
                                                                'combined_mult':combined_age_map_mri_mult[:,:,zz]})

                ''' >>> Part 2 <<< '''
                ''' Penalty + Global Normalisation (GN) '''
                combined_age_map_mri_normed = np.divide((combined_age_map_mri - np.min(combined_age_map_mri)),\
                                            (np.max(combined_age_map_mri) - np.min(combined_age_map_mri)))
                combined_age_map_mri_mult_normed = np.divide((combined_age_map_mri_mult - np.min(combined_age_map_mri_mult)),\
                                            (np.max(combined_age_map_mri_mult) - np.min(combined_age_map_mri_mult)))

                if save_jpeg:
                    for zz in range(0, mri_data.shape[2]):
                        fig2, axes2 = plt.subplots(1, 3)
                        fig2.set_size_inches(16,5)

                        axes2[0].set_title('Combined and normalised')
                        im1 = axes2[0].imshow(np.rot90(np.nan_to_num(combined_age_map_mri_normed[:,:,zz])), cmap="jet", vmin=0, vmax=1)
                        divider1 = make_axes_locatable(axes2[0])
                        cax1 = divider1.append_axes("right", size="7%", pad=0.05)
                        cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

                        axes2[1].set_title('Combined, penalised and normalised')
                        im2 = axes2[1].imshow(np.rot90(np.nan_to_num(combined_age_map_mri_mult_normed[:,:,zz])), cmap="jet", vmin=0, vmax=1)
                        divider2 = make_axes_locatable(axes2[1])
                        cax2 = divider2.append_axes("right", size="7%", pad=0.05)
                        cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

                        axes2[2].set_title('Original MRI slice')
                        im3 = axes2[2].imshow(np.rot90(np.nan_to_num(mri_data[:,:,zz])), cmap="gray")
                        divider3 = make_axes_locatable(axes2[2])
                        cax3 = divider3.append_axes("right", size="7%", pad=0.05)
                        cbar3 = plt.colorbar(im3, cax=cax3)

                        plt.tight_layout()
                        # Make space for title
                        plt.subplots_adjust(top=0.95)

                        ''' Save mri_data in *_combined.jpg '''
                        dirOutData = dirOutput + '/' + mri_code + '/IAM_combined_python/Combined/'
                        fig2.savefig(dirOutData + str(zz) + '_combined.jpg', dpi=100)
                        print('Saving files: ' + dirOutData + str(zz) + '_combined.jpg')
                        plt.close()

                ''' Save mri_data in *.mat '''
                sio.savemat(dirOutDataCom + '/all_slice_dat.mat', {'combined_age_map_all_slice':combined_age_map_mri,
                                                   'mri_slice_mul_all_slice':combined_age_map_mri_mult,
                                                   'combined_age_map_mri_normed':combined_age_map_mri_normed,
                                                   'combined_age_map_mri_mult_normed':combined_age_map_mri_mult_normed})

                '''
                combined_age_map_mri_img = nib.Nifti1Image(combined_age_map_mri_normed, mri_nii.affine)
                nib.save(combined_age_map_mri_img, str(dirOutDataFin + '/IAM_GPU_COMBINED.nii.gz'))

                combined_age_map_mri_GN_img = nib.Nifti1Image(combined_age_map_mri_mult_normed, mri_nii.affine)
                nib.save(combined_age_map_mri_GN_img, str(dirOutDataFin + '/IAM_GPU_GN.nii.gz'))
                '''

                ''' >>> Part 3 <<< '''
                ''' Post-processing '''
                ''' COMMENTED OUT BECAUSE NOT AVAILABLE
                if nawm_available and ~nawm_preprocessing:
                    combined_age_map_mri_mult_normed = np.multiply(combined_age_map_mri_mult_normed,nawm_mri_code)
                    combined_age_map_mri_GN_img = nib.Nifti1Image(combined_age_map_mri_mult_normed, mri_nii.affine)
                    nib.save(combined_age_map_mri_GN_img, str(dirOutDataFin + '/IAM_GPU_GN_postprocessed.nii.gz'))
                '''
                '''
                ---------------------------------------------------------------------
                KEY POINT: IAM's Combination, Penalisation, and Post-processing - END
                '''

                if delete_intermediary:
                    shutil.rmtree(dirOutDataCom, ignore_errors=True)
                    for xy in range(0, len(patch_size)):
                        shutil.rmtree(dirOutput + '/' + mri_code + '/' + str(patch_size[xy]), ignore_errors=True)

                del temp
                del center_source_patch, icv_source_flag
                del icv_source_flag_valid, index_mapping
                del area_source_patch, target_patches_np   # Free memory
                del area_source_patch_cuda_all, target_patches_np_cuda_all   # Free memory
                gc.collect()

        ## Print the elapsed time information
        print('\n--\nSpeed statistics of this run..')
        print('mean elapsed time  : ' + str(np.mean(elapsed_times_all)) + ' seconds')
        print('std elapsed time   : ' + str(np.std(elapsed_times_all)) + ' seconds')
        print('median elapsed time : ' + str(np.median(elapsed_times_all)) + ' seconds')
        print('min elapsed time   : ' + str(np.min(elapsed_times_all)) + ' seconds')
        print('max elapsed time   : ' + str(np.max(elapsed_times_all)) + ' seconds')
