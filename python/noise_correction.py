import argparse
import numpy as np
import torch
from utils import normalize, unnormalize
from gprgpu_toolkit import tune_gpr, GPModel, exact_gp, weighted_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type = str, required = True, help = "Filename to run the denoising")
    parser.add_argument("--sample_size", type = int, default = 500)
    parser.add_argument("--iter", type = int, default = 5)
    parser.add_argument("--pred_batch", type = bool, default = False)
    parser.add_argument("--predset", type = float)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    noisy_xyz = np.load(args.filename)
    normalized_noisy_xyz, norm_ind =  normalize(noisy_xyz)
    
    if args.pred_batch == True:
        batch_size = 2000 # need to be changed according to your memory capacity
    else:
        batch_size = False
    
    hyp_opt_2, tuned_model_2 = tune_gpr(device, normalized_noisy_xyz, 
                                        GPModel, n_sample = args.sample_size, 
                                        n_iter = args.iter, 
                                        min_iter = 3, convergence_tol = 1e-2, 
                                        early_stopping_patience = 3)
    if args.predset is not None:
        mean_all_2, std_all_2 = exact_gp(device, normalized_noisy_xyz, 
                                         GPModel, hyp_opt_2, n_sample = args.sample_size, 
                                         num_epochs = 2, pred_batch = args.pred_batch, 
                                         batch_size = batch_size, 
                                         pred_dataset = args.predset)
    else:
        mean_all_2, std_all_2 = exact_gp(device, normalized_noisy_xyz, 
                                         GPModel, hyp_opt_2, n_sample = args.sample_size, 
                                         num_epochs = 2, pred_batch = args.pred_batch, 
                                         batch_size = batch_size)
    
    weighted_prediction = weighted_pred(mean_all_2, std_all_2)
    normalized_xyz_corrected = np.column_stack((normalized_noisy_xyz[:,:2], weighted_prediction))
    xyz_corrected = unnormalize(normalized_xyz_corrected, norm_ind)
    
    np.save(args.filename.replace('.npy','_denoised.npy'), xyz_corrected)
    
if __name__ == "__main__":
    main()
    
    