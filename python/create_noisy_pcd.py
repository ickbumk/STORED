""" Syntheticall generate the noisy outlier dataset. In your command line run:
    "python create_noisy_pcd.py --filename [your_file_name] --percentage [your_percentage] --multiplier [your_multiplier]"
    The output filename should be '[your_file_name]_noisy.npy', the 3 column xyz file and '[your_file_name]_indices.npy', the indices for the outliers.
"""

import argparse
import numpy as np
from utils import add_outlier_and_noise_mean

def main():
    parser = argparse.ArgumentParser(description = "Add Outlier and Noise to the given data")
    parser.add_argument("--filename", type= str, required = True, help = "Name of the input file")
    parser.add_argument("--percentage", type = int, default = 2, help = "Percentage of outliers")
    parser.add_argument("--multiplier", type = float, default = 0.5, help = "Gaussian noise truncation bounds determined based on the median absolute deviation of the given point cloud")
    
    
    args = parser.parse_args()
    input_file = args.filename
    noisy_file = input_file.replace('.npy', '_noisy.npy')
    outlier_file = input_file.replace('.npy','_indices.npy')
    
    xyz = np.load(input_file)        # it has to be .npy file
    z_outlier, outlier_indices = add_outlier_and_noise_mean(xyz, mad_multiplier = args.multiplier, percentage_outlier = args.percentage)
    
    xyz_outlier = np.column_stack((xyz[:,:2],z_outlier))
    
    np.save(noisy_file, xyz_outlier)
    np.save(outlier_file, outlier_indices)
    
    
if __name__ == "__main__":
    main()