import argparse
import numpy as np
from utils import determine_best_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type = str, required = True, help = "The file must have _noisy at the end")
    parser.add_argument("--score", type = int, default = 0)
    
    args = parser.parse_args()
    
    filename = args.filename
    
    if filename.endswith("_noisy.npy") == False:
        raise ValueError("Invalid file name. Filename must end with '_noisy.npy'")
    
    threshold = args.score
    
    noisy_xyz = np.load(filename)
    noisy_score = np.load(filename.replace('_noisy.npy','_outlier_score.npy'))
    outlier_indices = np.load(filename.replace('_noisy.npy','_indices.npy'))
    
    
    if threshold == 0:
        print("Threshold not specified by the user, finding the optimal threshold...")
        best_threshold, f1, precision, recall = determine_best_t(noisy_score, outlier_indices) 
        outlier_pred = noisy_score > best_threshold
        xyz_clean = noisy_xyz[~outlier_pred]
        np.save(filename.replace('.npy','_clean.npy'), xyz_clean)
        print('-----------')
        print('-----------')
        print('-----------')
        print('F1 score is:', f1)
        print('Recall is:', precision)
        print('Precision is:', recall)
        print('-----------')
        print('-----------')
        print('-----------')
    else:
        print(f'User specified the threshold...{threshold}')
        outlier_pred = noisy_score > threshold
        xyz_clean = noisy_xyz[~outlier_pred]
        np.save(filename.replace('.npy','_clean.npy'), xyz_clean)
        
if __name__ == "__main__":
    main()