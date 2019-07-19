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
import code
import re
from timeit import default_timer as timer

def jaccard_index(prediction, ground_truth):
    sum_images = prediction + ground_truth
    jn = np.count_nonzero(sum_images == 2)
    jd = np.count_nonzero(sum_images > 0)

    return jn/jd


def dice_coefficient(jaccard_index):
    return (2*jaccard_index) / (1+jaccard_index)

def sensitivity(TP, FN):
    return TP/(TP + FN)

def positive_predictive_value(TP, FP):
    return TP/(TP+FP)

def specificity(FP, TN):
    return TN/(TN + FP)


def damage_metric(thresholded, roi, original_mri):
    original_mri = np.nan_to_num(original_mri)
    nawm = np.nan_to_num(roi - thresholded)

    wmh_intensity = np.sum(original_mri[thresholded == 1]) / np.count_nonzero(thresholded)
    nawm_intensity = np.sum(original_mri[nawm == 1]) / np.count_nonzero(nawm)

    wmh_volume = np.count_nonzero(thresholded)
    nawm_volume = np.count_nonzero(nawm)

    damage_metric = (wmh_intensity - nawm_intensity)/nawm_intensity
    damage_metric *= (wmh_volume / (wmh_volume + nawm_volume))

    return damage_metric

def get_true_false_positive_negative(prediction, ground_truth, roi):
    ground_truth = ground_truth.astype(float)
    prediction = prediction.astype(float)
    ground_truth[np.isnan(roi)] = np.nan
    prediction[np.isnan(roi)] = np.nan


    TP = sum(prediction[ground_truth == 1])
    FP = sum(prediction[ground_truth == 0])

    prediction = 1 - prediction
    TN = sum(prediction[ground_truth == 0])
    FN = sum(prediction[ground_truth == 1])

    return TP, FP, TN, FN



def get_volume(thresholded):
    return np.count_nonzero(thresholded)

def calculate_mean_and_difference(prediction, ground_truth):
    pred_vol = get_volume(prediction)
    gt_vol = get_volume(ground_truth)
    return (pred_vol + gt_vol) / 2, (pred_vol - gt_vol)


def plot_altman_bland(mean_difference, exact_name_result_folder):
    #np.save("mean_difference.npy", np.asarray(mean_difference))
    x = np.asarray(mean_difference)[:, 0]
    y = np.asarray(mean_difference)[:, 1]
    print(x)
    print(y)
    plt.scatter(x, y)

    ## Mean
    plt.hlines(np.mean(y), xmin=0, xmax=np.max(x), linestyles='solid', colors='blue')

    ## Confidence
    plt.hlines(np.mean(y) + 1.96*np.std(y), xmin=0, xmax=np.max(x), linestyles='dashed', colors='red')
    plt.hlines(np.mean(y) - 1.96*np.std(y), xmin=0, xmax=np.max(x), linestyles='dashed', colors='red')

    plt.grid()
    plt.show()
    plt.savefig(exact_name_result_folder + ".jpg")

def plot_average_dice(specific_results_folder, threshold_list, dice_list):
    plt.plot(threshold_list, dice_list)
    plt.ylim([0,1])
    plt.title(specific_results_folder)
    plt.xlabel("Threshold level")
    plt.ylabel("Dice averages")
    plt.show()



def create_csv(exact_name_result_folder):
    with open('D:/Edinburgh/dissertation/evaluations/evaluation_' + exact_name_result_folder + '.csv', mode='w') as evaluation:
        evaluation = csv.writer(evaluation, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
        evaluation.writerow(['code', 'version', 'threshold', "pred_volume", "gt_volume", "TP", "FP", "TN", "FN", 'jaccard', 'dice', 'positive_predictive_value', 'sensitivity', 'specificity', 'bland_altman_mean', 'bland_altman_difference', 'damage_metric_pred', 'damage_metric_gt'])

def write_scan_to_csv(exact_name_result_folder, patient_code, scan_number, THRSH, prediction, ground_truth, roi, original_mri):
    with open('D:/Edinburgh/dissertation/evaluations/evaluation_' + exact_name_result_folder + '.csv', 'a') as evaluation:
        evaluation = csv.writer(evaluation, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')

        ## Compute metrics
        pred_volume = get_volume(prediction)
        gt_volume = get_volume(ground_truth)
        TP, FP, TN, FN = get_true_false_positive_negative(prediction, ground_truth, roi)
        jaccard = jaccard_index(prediction, ground_truth)
        dice = dice_coefficient(jaccard)
        pos_pred_val = positive_predictive_value(TP, FP)
        sens = sensitivity(TP, FN)
        spec = specificity(FP, TN)
        bland_altman_mean, bland_altman_difference = calculate_mean_and_difference(prediction, ground_truth)
        damage_metric_pred = damage_metric(prediction, roi, original_mri)
        damage_metric_gt = damage_metric(ground_truth, roi, original_mri)



        ## Print metrics
        print("prediction volume:", pred_volume)
        print("ground truth volume:", gt_volume)
        print("TP/FP/TN/FN:", TP, FP, TN, FN)
        print("jaccard:", jaccard)
        print("dice:", dice)
        print("positive_predictive_value:", pos_pred_val)
        print("sensitivity:", sens)
        print("specificity:", spec)
        print("bland_altman_mean:", bland_altman_mean)
        print("bland_altman_difference:", bland_altman_difference)
        print("damage_metric pred:", damage_metric_pred)
        print("damage_metric gt:", damage_metric_gt)

        row = [patient_code, scan_number, THRSH, pred_volume,
                gt_volume, TP, FP, TN, FN, jaccard, dice,
                pos_pred_val, sens, spec, bland_altman_mean,
                bland_altman_difference, damage_metric_pred,
                damage_metric_gt]

        evaluation.writerow(row)



def evaluate(THRSH, specific_results=None):

    ## Set the paths
    path_data = "D:/edinburgh/dissertation/data/"
    path_results = "D:/edinburgh/dissertation/"


    ## Get results directories
    pattern = "results*"
    results_dirs = [pathlib.PurePath(path_results, directory) for directory in next(os.walk(path_results))[1] if fnmatch(directory, pattern)]

    ## Not all ground truth are available
    no_ground_truth_available = ["DMP01", "DMP02", "DMP03", "DMP04", "DMP05",
                                "DMP06", "DMP07", "DMP08", "DMP09", "DMP10",
                                "DMP11", "DMP12", "DMP13", "DMP14", "DMP15",
                                "DMP16", "DMP17"]


    ## Get different sample number directories
    for results_directory in results_dirs:
        ## List to keep tuples for Bland-Altman analysis
        mean_difference = []

        pattern = "_*"
        results_directory = str(results_directory)

        iterator = re.search(r'results', results_directory)
        exact_name_result_folder = results_directory[iterator.start():]

        diff_sample_dirs = [pathlib.PurePath(results_directory, directory) for directory in next(os.walk(results_directory))[1] if fnmatch(directory, pattern)]

        ## Get patients directories
        pattern = "DMP*"
        for diff_sample in diff_sample_dirs:
            if specific_results != None and specific_results != exact_name_result_folder:
                break

            ## Create CSV if it does not exist yet
            if not os.path.exists(path_results + '/evaluations/evaluation_' + exact_name_result_folder + '.csv'):
                create_csv(exact_name_result_folder)

            print("NOW EVALUATING FOLDER:", exact_name_result_folder, "with threshold:", THRSH)
            diff_sample = str(diff_sample)
            patient_folders = [pathlib.PurePath(diff_sample, directory) for directory in next(os.walk(diff_sample))[1] if fnmatch(directory, pattern)]

            ## Get scan versions
            pattern = "V*"
            for scan_version in patient_folders:
                scan_version = str(scan_version)
                if scan_version[-5:] in no_ground_truth_available:
                    print("Ground truth not available for", scan_version[-5:])
                    continue
                scans = [pathlib.PurePath(scan_version, directory) for directory in next(os.walk(scan_version))[1] if fnmatch(directory, pattern)]

                ## Get in the version directory
                for scan in scans:
                    scan = str(scan)
                    print("\nProcessing patient " + scan_version[-5:] + ", " + scan[-2:], "From", exact_name_result_folder, "THRSH:", THRSH)

                    ## Load prediction given by LOTS-IAM (2D folders has a diff. structure)
                    if results_directory[-2:] == "2d":
                        prediction = sio.loadmat(scan + "/IAM_GPU_nifti_python/all_slice_dat.mat")
                    else:
                        prediction = sio.loadmat(scan + "/IAM_combined_python/all_slice_dat.mat")

                    prediction = prediction["combined_age_map_mri_mult_normed"]

                    ## Threshold prediction
                    prediction[prediction > THRSH] = 1
                    prediction[prediction < THRSH] = 0

                    ## Load original MRI
                    original_mri = sio.loadmat("D:/Edinburgh/dissertation/data/" + scan_version[-5:] + "/" + scan[-2:] + "/flair.mat")
                    original_mri = original_mri["flair"]

                    ## Load ground truth
                    ground_truth = sio.loadmat(path_data + scan_version[-5:] + "/" +  scan[-2:] + "/ground_truth.mat")["WMHmask_data"]

                    ## Get ROI
                    roi = original_mri.copy()
                    roi[~np.isnan(roi)] = 1

                    ## Write scan to CSV
                    write_scan_to_csv(exact_name_result_folder, scan_version[-5:], scan[-2:], THRSH, prediction, ground_truth, roi, original_mri)

            print("FINISHED ALL PATIENTS IN", exact_name_result_folder)
