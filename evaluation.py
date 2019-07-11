"""
This file calculates the Jaccard index, Dice coefficient, Positive Predictive Value, Sensitivity and Specificity according to
the ground truth. Then, it outputs a csv file containing all the information.
This assumes binary arrays for both prediction and ground truth. This also assumes the roi to have values of 1s and nans.
"""

import numpy as np
import csv
import sys
import os, pathlib
from fnmatch import fnmatch
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

def jaccard(prediction, ground_truth):
    sum_images = prediction + ground_truth
    jn = np.count_nonzero(sum_images == 2)
    jd = np.count_nonzero(sum_images > 0)

    return jn/jd


def dice_coefficient(jaccard_index):
    return (2*jaccard_index) / (1+jaccard_index)


def sensitivity(prediction, ground_truth):
    sum_images = prediction + ground_truth
    tp = np.count_nonzero(sum_images == 2)
    prediction = 1 - prediction
    td = sum(prediction[ground_truth == 1])
    TPF = tp/(tp+td)

    return TPF

def positive_predictive_value(prediction, ground_truth):
    sum_images = prediction + ground_truth
    pn = np.count_nonzero(sum_images == 2)
    pd = sum(prediction[ground_truth == 0])
    TPF = pn/(pn+pd)

    return TPF

def specificity(prediction, ground_truth, roi):
    ground_truth[np.isnan(roi)] = np.nan
    prediction[np.isnan(roi)] = np.nan
    sn = np.count_nonzero(ground_truth == 0)
    sd = sum(prediction[ground_truth == 0])
    return sn/(sn+sd)


def damage_metric(thresholded, roi):
    thresholded[thresholded == 0] = np.nan
    nawm = np.nan_to_num(roi - thresholded)

    wmh_intensity = sum(thresholded[~np.isnan(thresholded)]) / wmh.count_nonzero(~np.isnan(thresholded))
    nawm_intensity = sum(nawm) / np.count_nonzero(~np.isnan(nawm))

    wmh_volume = np.count_nonzero(~np.isnan(thresholded))
    nawm_volume = np.count_nonzero(~np.isnan(nawm))

    damage_metric = (wmh_intensity - nawm_intensity)/nawm_intensity
    damage_metric *= (wmh_volume / (wmh_volume + nawm_volume))

    return damage_metric


## Set the paths
path_data = "W:/"
path_results = "W:/"
if sys.argv[1] == "melvin":
    path_data = "D:/edinburgh/dissertation/data/"
    path_results = "D:/edinburgh/dissertation/"


## Get results directories
pattern = "results*"
results_dirs = [pathlib.PurePath(path_results, directory) for directory in next(os.walk(path_results))[1] if fnmatch(directory, pattern)]

## Get different sample number directories
for results_directory in results_dirs:
    pattern = "_*"
    results_directory = str(results_directory)
    print(results_directory)
    diff_sample_dirs = [pathlib.PurePath(results_directory, directory) for directory in next(os.walk(results_directory))[1] if fnmatch(directory, pattern)]

    ## Get patients directories
    pattern = "DMP*"
    for diff_sample in diff_sample_dirs:
        if results_directory[-2:] == "2d":
            break
        diff_sample = str(diff_sample)
        patient_folders = [pathlib.PurePath(diff_sample, directory) for directory in next(os.walk(diff_sample))[1] if fnmatch(directory, pattern)]

        ## Get scan versions
        pattern = "V*"
        for scan_version in patient_folders:
            scan_version = str(scan_version)
            scans = [pathlib.PurePath(scan_version, directory) for directory in next(os.walk(scan_version))[1] if fnmatch(directory, pattern)]

            ## Get in the version directory
            for scan in scans:
                scan = str(scan)
                print("Processing patient " + str(scan_version[-5:]) + ", " + scan[-2:])

                ## Load prediction given by LOTS-IAM
                prediction = sio.loadmat(scan + "\\IAM_combined_python\\all_slice_dat.mat")
                prediction = prediction["combined_age_map_mri_mult_normed"]

                ## Load ground truth
                #ground_truth = sio.loadmat(path_data + ".mat")
            raise Exception()







with open('evaluation.csv', mode='w') as evaluation:
    evaluation = csv.writer(evaluation, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    evaluation.writerow(['code', 'version', 'jaccard', 'dice', 'positive_predictive_value', 'sensitivity', 'specificity', 'damage_metric_real', 'damage_metric_pred'])
    evaluation.writerow(['DMP01', 'V1', '0.2', '0.232', '342342', '23423', '23423423', '123456', '4510.04'])
