# Download NCAR tornado dataset
import sys
import tarfile
import time
import copy
import glob
import os
import os.path
import numpy as np
import netCDF4
from urllib.request import urlretrieve
from optparse import OptionParser

# Definitions
online_image_file_name = (
    'https://storage.googleapis.com/track_data_ncar_ams_3km_nc_small/'
    'track_data_ncar_ams_3km_nc_small.tar.gz')
target_name = ("RVORT1_MAX_future")
predictor_names = ["REFL_COM_curr", "T2_curr", "U10_curr", "V10_curr"]

# Options
parser = OptionParser()
parser.add_option("-o", "--out_dir", default="tornado/data/", help="Path to download data.")
parser.add_option("-n", "--num_files", default=25, type=int, help="Save only first N files.")
(options, args) = parser.parse_args()

out_dir = options.out_dir
n_files = options.num_files

# Download data
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
local_file = out_dir + "/" + os.path.split(online_image_file_name)[-1]
t0 = time.time()
print('Downloading data to: "{0:s}"...'.format(local_file))
urlretrieve(online_image_file_name, local_file)
print('Unzipping file: "{0:s}"...'.format(local_file))
tar_file_handle = tarfile.open(local_file)
tar_file_handle.extractall(out_dir)
tar_file_handle.close()
print("Downloaded and unzipped in {} seconds".format(time.time() - t0))
img_dir = out_dir + "/" + os.path.split(online_image_file_name)[-1].split(".")[0]
files = sorted(glob.glob(img_dir + "/*.nc"))
print("Found {} netCDF files: ".format(len(files)))
files = files[:n_files]
print("Processing first {} files".format(n_files))

bands = len(predictor_names)
rows = 32
cols = 32

# Get number of samples in each file
sample_sizes = [netCDF4.Dataset(nc_file).variables["track_step"][:].shape[0] for nc_file in files]
# Init storage
predictors = [np.zeros((samples, rows, cols, bands)).astype(float) for samples in sample_sizes]
targets = [np.zeros((samples, rows, cols)).astype(float) for samples in sample_sizes]
# Loop over files, getting the predictors & targets
for nidx, nc_file in enumerate(files):
	nc = netCDF4.Dataset(nc_file)
	targets[nidx][:, :] = nc.variables[target_name][:]
	for bidx, predictor_name in enumerate(predictor_names):
		predictors[nidx][:, :, :, bidx] = nc.variables[predictor_name][:]
predictors = np.concatenate(predictors)
targets = np.concatenate(targets)

# Target binarization
percentile_level = 90
max_target_values = np.max(targets, axis=(1, 2))
binarization_threshold = np.percentile(max_target_values, percentile_level)
target_values = np.array(np.max(targets, axis=(1,2)) \
                         >= binarization_threshold).astype(int)

# Normalize predictors
predictors_norm = np.zeros(predictors.shape, dtype=float)
scale_cols = ["mean", "std"]
for i in range(bands):
  mean = predictors[:, :, :, i].mean()
  std = predictors[:, :, :, i].std()
  predictors_norm[:, :, :, i] = \
    (predictors[:, :, :, i] - mean) / std

# Save data
out_file = out_dir + "/dataset.npz"
np.savez(out_file, X = predictors_norm, y = target_values)
print("Saved data to: {}".format(out_file))
print("  'X' : {}".format(predictors_norm.shape))
print("  'y' : {}".format(target_values.shape))
