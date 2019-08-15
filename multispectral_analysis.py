import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import code, csv, sys, os
from scipy.misc import imread, imshow
from sklearn.datasets import load_sample_image
from sklearn.mixture import GaussianMixture
from visualization import visualize_3d_gmm
import operator

# code.interact(local=dict(globals(), **locals()))

def get_average_intensity_from_clusters(clusters, original_mri):
    return {str(i):np.sum(original_mri[clusters == i])/np.count_nonzero(clusters == i) for i in np.unique(clusters)}


first_dataset = ["DMP0" + str(scan_code) for scan_code in np.arange(1, 10)] + ["DMP" + str(scan_code) for scan_code in np.arange(10, 18)]
second_dataset = ["DMP" + str(scan_code) for scan_code in np.arange(18, 61)]
scan_versions = ["V" + str(version) for version in np.arange(1, 4)]

num_of_clusters = 4

for scan_code in first_dataset:
    for scan_version in scan_versions:

        ## load probability maps from LOTS-IAM-3D
        start_path = 'D:/Edinburgh/dissertation/'
        end_path = 'IAM_combined_python/all_slice_dat'
        #t1 = sio.loadmat(start_path + '/results_3d_t1/_64s16m/' +  scan_code + '/' + scan_version + '/' + end_path)
        t2 = sio.loadmat(start_path + '/results_3d_t2/_64s16m/' +  scan_code + '/' + scan_version + '/' + end_path)
        flair = sio.loadmat(start_path + '/results_3d/_64s16m/' +  scan_code + '/' + scan_version + '/' + end_path)

        #t1 = t1["combined_age_map_mri_mult_normed"]
        t2 = t2["combined_age_map_mri_mult_normed"]
        flair = flair["combined_age_map_mri_mult_normed"]

        ''' I'm trying to comment this out but I don't know why it would be better
        ## Converting the "0's" to NaN
        t1[t1 == np.min(t1)] = np.nan
        t2[t2 == np.min(t2)] = np.nan
        flair[flair == np.min(flair)] = np.nan
        '''
        ## Reshape to comply with scikit-learn's data representation
        X = np.array([t2.flatten(), flair.flatten()])
        X = np.rot90(X)



        ## Create Gaussian Mixture Model
        gmm = GaussianMixture(covariance_type='full', n_components=num_of_clusters)
        gmm.fit(X)
        clusters = gmm.predict(X)
        clusters = clusters.reshape(256, 256, 176)
        clusters = np.flip(clusters)

        avg_intensities_per_clusters = get_average_intensity_from_clusters(clusters, flair)
        WMH_cluster_number = max(avg_intensities_per_clusters.items(), key=operator.itemgetter(1))[0]


        clusters = (clusters == WMH_cluster_number)

        sio.savemat("D:/Edinburgh/dissertation/results_clustering_t2_flair/" + scan_code + "_" + scan_version + ".mat")




        #visualization.visualize_3d_gmm(X, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
