import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import code, csv, sys, os
from scipy.misc import imread, imshow
from sklearn.datasets import load_sample_image
from sklearn.mixture import GaussianMixture
from visualization import visualize_3d_gmm


first_dataset = ["DMP0" + str(code) for code in np.arange(1, 10)] + ["DMP" + str(code) for code in np.arange(10, 18)]
second_dataset = ["DMP" + str(code) for code in np.arange(18, 61)]
scan_versions = ["V" + str(version) for version in np.arange(1, 4)]

num_of_clusters = 4

for code in first_dataset:
    for scan_version in scan_versions:

        ## load probability maps from LOTS-IAM-3D
        start_path = 'D:/Edinburgh/dissertation/'
        end_path = 'IAM_combined_python/all_slice_dat'
        t1 = sio.loadmat(start_path + '/results_t1/_64s16m/' +  code + '/' + scan_version + '/' + end_path)
        t2 = sio.loadmat(start_path + '/results_t2/_64s16m/' +  code + '/' + scan_version + '/' + end_path)
        flair = sio.loadmat(start_path + '/results_3d/_64s16m/' +  code + '/' + scan_version + '/' + end_path)

        t1 = t1["combined_age_map_mri_mult_normed"]
        t2 = t2["combined_age_map_mri_mult_normed"]
        flair = flair["combined_age_map_mri_mult_normed"]

        ''' I'm trying to comment this out but I don't know why it would be better
        ## Converting the "0's" to NaN
        t1[t1 == np.min(t1)] = np.nan
        t2[t2 == np.min(t2)] = np.nan
        flair[flair == np.min(flair)] = np.nan
        '''
        ## Reshape to comply with scikit-learn's data representation
        X = np.array([t1.flatten(), t2.flatten(), flair.flatten()])

        ## Create Gaussian Mixture Model
        gmm = GaussianMixture(covariance_type='full', n_components=num_of_clusters)
        gmm.fit(X)
        clusters = gmm.predict(X)
        clusters = clusters.reshape(256, 256)
        imshow(clusters)

        visualization.visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
