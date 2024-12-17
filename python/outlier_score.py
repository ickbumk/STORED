import argparse
import numpy as np
import torch
from utils import normalize
from gprgpu_toolkit import tune_gpr, GPModel, exact_gp, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type = str, required = True)
    parser.add_argument("--sample_size", type = int, default = 500)
    parser.add_argument("--iter", type = int, default = 100)
    
    args = parser.parse_args()
    noisy_file = args.filename
    n_sample = args.sample_size
    n_iter = args.iter
    
    xyz_outlier = np.load(noisy_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    normalized_xyz_outlier, norm_ind = normalize(xyz_outlier)
    hyp_opt, tuned_model = tune_gpr(device, normalized_xyz_outlier, GPModel, 
                                    n_sample = n_sample, n_iter = n_iter, min_iter = 10, convergence_tol = 1e-2, early_stopping_patience = 3)
    mean_all, std_all = exact_gp(device, normalized_xyz_outlier, GPModel, 
                                 hyp_opt, n_sample = n_sample, num_epochs = 2, pred_batch = True, batch_size = False, pred_dataset = [])
    score_final = score(mean_all, std_all, normalized_xyz_outlier)
    
    np.save(noisy_file.replace('_noisy.npy','_outlier_score.npy'), score_final)
    
if __name__ == "__main__":
    main()