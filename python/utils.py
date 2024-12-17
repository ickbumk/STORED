import numpy as np
import re
import os
import cv2
import open3d as o3d
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.spatial import cKDTree
from scipy.stats import truncnorm
from matplotlib import pyplot as plt

def chamfer_distance(point_cloud1, point_cloud2):
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)
    distances1, _ = tree1.query(point_cloud2)
    distances2, _ = tree2.query(point_cloud1)
    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    
    return chamfer_dist

def normalize(xyz):
    original_mean = np.mean(xyz, axis = 0)
    original_std = np.std(xyz, axis = 0)
    normalized_xyz = (xyz -original_mean)/original_std
    return normalized_xyz, np.vstack((original_mean, original_std))

def unnormalize(xyz, norm_ind):
    norm_mean, norm_std = norm_ind
    unnorm_xyz = xyz*norm_std+norm_mean
    return unnorm_xyz
    

def visualize(xyz):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([point_cloud])

def visualize_two(xyz_1, xyz_2):
    
    red_color = np.array([1, 0, 0])  # RGB for red
    red_colors = np.tile(red_color, (xyz_1.shape[0], 1))
    
    blue_color = np.array([0, 0, 1])  # RGB for red
    blue_colors = np.tile(blue_color, (xyz_2.shape[0], 1))
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.vstack((xyz_1,xyz_2)))
    point_cloud.colors = o3d.utility.Vector3dVector(np.vstack((red_colors,blue_colors)))
    o3d.visualization.draw_geometries([point_cloud])

def add_outlier_and_noise_mean(xyz, percentage_outlier=2, mad_multiplier = 0.5):

    bounding_box_diag = np.sqrt(np.sum((np.max(xyz, axis = 0) - np.min(xyz, axis = 0))**2))
    
    z = xyz[:,2]

    mad = np.median(np.abs(z-np.median(z)))
    
    min_outlier = mad_multiplier*mad

    print('MAD is:', mad)
    
    n_outliers = 2*int(percentage_outlier/100 * z.shape[0]/2)  #upper and lower
    n_noise = z.shape[0]-n_outliers

    if n_outliers == 0:
        total_outlier = 0
    else:
        half_outliers = np.random.normal(0, 0.2*bounding_box_diag, int(n_outliers/2))
        half_outliers = np.abs(half_outliers)+min_outlier
        # half_outliers -= half_outliers.min()-min_outlier
        total_outlier = np.hstack((half_outliers,-half_outliers))
        random.shuffle(total_outlier)

    s = 0.01*bounding_box_diag
    
    a, b = (-min_outlier) / s, (min_outlier) / s
    noise_normal = truncnorm(a,b, loc=0, scale = s)
    truncated_gaussian_noise =  noise_normal.rvs(size = n_noise)
    
    outlier_indices = random.sample(range(z.shape[0]),n_outliers)
    
    mask = np.zeros_like(z, dtype=bool)
    mask[outlier_indices] = True
    
    z_n = z.copy()
    z_n[mask]+=total_outlier
    z_n[~mask]+=truncated_gaussian_noise

    # plt.hist(np.hstack((truncated_gaussian_noise, total_outlier)), bins = 100)
    # plt.show()

    return z_n, mask

def determine_best_t(score_final, outlier_indices):
    score_list = np.unique(score_final)
    f1_list = []
    p_list = []
    r_list = []
    for score_iter in score_list:
        detected_outliers = score_final > score_iter
        f1_list.append(f1_score(outlier_indices, detected_outliers))
        p_list.append(precision_score(outlier_indices, detected_outliers, zero_division = 0.0))
        r_list.append(recall_score(outlier_indices, detected_outliers))

    best_f1_ind = np.argmax(f1_list)
    
    best_t = score_list[best_f1_ind]
    

    return best_t, np.max(f1_list), p_list[best_f1_ind], r_list[best_f1_ind]