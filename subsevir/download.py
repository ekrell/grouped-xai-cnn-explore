# Download sub-sevir dataset
# https://www.ai2es.org/data/dataset-details-sub-sevir/ 

# Notes:
# - 'ML-ready': already normalized
# - Regression label: number of flashes
# - Classification label: if any flashes

import sys
import os
import tarfile
import time
from urllib.request import urlretrieve
from optparse import OptionParser
import numpy as np
import xarray as xr
from scipy import ndimage

# Definitions
url = "https://zenodo.org/record/7011372/files/sub-sevir.tar.gz?download=1"

# Options
parser = OptionParser()
parser.add_option("-o", "--out_dir", default="subsevir/data/", help="Path to download data")
parser.add_option("-w", "--width", default=32, type="int", help="Width in grid cells.")
(options, args) = parser.parse_args()

out_dir = options.out_dir
width = options.width

# Download data
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
local_file = f"{out_dir}/sub-sevir.tar.gz"
t0 = time.time()
print('Downloading data to: "{0:s}"...'.format(local_file))
urlretrieve(url, local_file)
files = [local_file, 
  f"{out_dir}/sub-sevir/sub-sevir-train.tar.gz",
  f"{out_dir}/sub-sevir/sub-sevir-val.tar.gz",
  f"{out_dir}/sub-sevir/sub-sevir-test.tar.gz"]
for file in files:
  print('Unzipping file: "{0:s}"...'.format(file))
  tar_file_handle = tarfile.open(file)
  tar_file_handle.extractall(out_dir)
  tar_file_handle.close()
print("Downloaded and unzipped in {} seconds".format(time.time() - t0))


datasets = [
  (f"{out_dir}/sub-sevir-train.zarr", f"{out_dir}/dataset_train.npz"),
  (f"{out_dir}/sub-sevir-val.zarr",   f"{out_dir}/dataset_valid.npz"),
  (f"{out_dir}/sub-sevir-test.zarr",  f"{out_dir}/dataset_test.npz")]

for dataset in datasets:
  # Extract data  
  ds = xr.open_dataset(dataset[0], engine="zarr")
  # Raster features
  X = ds.features.values
  # Resize 
  cols = X.shape[2]
  scale_prop = float(width) / float(cols)
  X = np.array([ndimage.zoom(x, (scale_prop, scale_prop, 1), order=3) for x in X])
  # Targets (classification)
  y = ds.label_1d_class.values
  # Targets (regression)
  y_regr = ds.label_1d_reg.values

  # Save
  np.savez(dataset[1], X=X, y=y, y_regr=y_regr)
  print(f"Saved data to: {dataset[1]}")
  print(f"  'X' : {X.shape}")
  print(f"  'y' : {y.shape}")
  print(f"  'y_regr' : {y_regr.shape}")
