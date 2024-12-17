import torch
import numpy as np
import gc
import glob
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import *
from gprgpu_toolkit import *

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = np.load('bun000.npy')
    x_original, y_original, z_original = xyz.T
    
    z_outlier, outlier_indices = add_outlier_and_noise_mean(xyz, mad_multiplier = 0.5, percentage_outlier = 5)
    
    xyz_outlier = np.column_stack((x_original, y_original, z_outlier))
    
    normalized_xyz_outlier, norm_ind = normalize(xyz_outlier)
    
    hyp_opt, tuned_model = tune_gpr(device, normalized_xyz_outlier, GPModel, n_sample = 500, n_iter = 100, min_iter = 10, convergence_tol = 1e-2, early_stopping_patience = 3)
    mean_all, std_all = exact_gp(device, normalized_xyz_outlier, GPModel, hyp_opt, n_sample = 500, num_epochs = 2, pred_batch = True, batch_size = False, pred_dataset = [])
    score_final = score(mean_all, std_all, normalized_xyz_outlier)
    
    # Determine the best threshold (if ground truth not provided, then the user must define)
    best_threshold, f1, precision, recall = determine_best_t(score_final, outlier_indices)
    
    # Clear Memory
    torch.cuda.empty_cache()
    gc.collect()
    
    outlier_pred = score_final>best_threshold
    
    print('-----------')
    print('-----------')
    print('-----------')
    print('F1 score is:', f1)
    print('Recall is:', precision)
    print('Precision is:', recall)
    print('-----------')
    print('-----------')
    print('-----------')
    
    
    weighted_prediction = weighted_pred(mean_all, std_all)
    normalized_xyz_corrected = np.column_stack((normalized_xyz_outlier[:,:2],weighted_prediction))
    xyz_corrected = unnormalize(normalized_xyz_corrected, norm_ind)
    
    normalized_without_pred_outliers = normalized_xyz_outlier[~outlier_pred]
    
    # Recommended to tune_gpr with low iterations for better noise correction result
    hyp_opt_2, tuned_model_2 = tune_gpr(device, normalized_without_pred_outliers, GPModel, n_sample = 500, n_iter = 5, min_iter = 3, convergence_tol = 1e-2, early_stopping_patience = 3)
    mean_all_2, std_all_2 = exact_gp(device, normalized_without_pred_outliers, GPModel, hyp_opt_2, n_sample = 500, num_epochs = 2, pred_batch = False, batch_size = False, pred_dataset = normalized_xyz_outlier)
    weighted_prediction = weighted_pred(mean_all_2, std_all_2)
    
    
    # Clear Memory
    torch.cuda.empty_cache()
    gc.collect()
    
    normalized_xyz_corrected = np.column_stack((normalized_xyz_outlier[:,:2],weighted_prediction))
    
    xyz_corrected = unnormalize(normalized_xyz_corrected, norm_ind)
    
    print('-----------')
    print('-----------')
    print('-----------')
    print('Noise correction complete with Chamfer Distance decrease (%):', 100*(1-chamfer_distance(xyz, xyz_corrected)/chamfer_distance(xyz,xyz_outlier)))
    print('-----------')
    print('-----------')
    print('-----------')

    # visualize(xyz_corrected) #comment out if you don't want visualization

    # visualize_two(xyz_corrected, xyz)  #comment out if you don't want visualization
    
if __name__ == "__main__":
    main()