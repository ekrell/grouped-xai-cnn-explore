# Download EuroSAT dataset
# https://github.com/phelber/EuroSAT

import sys
import os
import time
from urllib.request import urlretrieve
from optparse import OptionParser
import numpy as np
import zipfile
import glob
from skimage import io
import random

# Definitions
url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"

classes = [
  ("AnnualCrop", 0),
  #("Forest", 0),
  #("HerbaceousVegetation", 0),
  ("Highway", 1),
  ("Industrial", 1),
  ("Pasture", 0),
  ("PermanentCrop", 0),
  ("Residential", 1),
  #("River", 0),
  #("SeaLake", 0),
]
n_classes = len(classes)

# Options
parser = OptionParser()
parser.add_option("-o", "--out_dir", default="eurosat/data/", help="Path to download data.")
(options, args) = parser.parse_args()

out_dir = options.out_dir

nc = 1000

# Download data
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
local_file = f"{out_dir}/EuroSATallBands.zip"
t0 = time.time()
print('Downloading data to: "{0:s}"...'.format(local_file))
#urlretrieve(url, local_file)
#with zipfile.ZipFile(local_file, 'r') as zip_ref:
#  zip_ref.extractall(out_dir)
print("Downloaded and unzipped in {} seconds".format(time.time() - t0))

img_dir_pre = f"{out_dir}/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/"

targets = [None for c in range(n_classes)]
features = [None for c in range(n_classes)]
for c in range(n_classes):
  img_dir = img_dir_pre + classes[c][0]

  files = glob.glob(img_dir + "/*tif")[:nc]
  n_files = len(files)
  
  targets[c] = np.zeros(n_files).astype("int")
  targets[c][:] = classes[c][1]
  features[c] = np.zeros((n_files, 64, 64, 13))

  for i, f in enumerate(files):
    features[c][i] = io.imread(f)

  print("  Images: ", classes[c][0])
  print("  Label:  ", classes[c][1])
  print("  Shape:  ", features[c].shape)
  
y = np.concatenate(targets)
X = np.concatenate(features)

# Normalize
means = np.mean(X, axis=(0, 1, 2))
stds = np.std(X, axis=(0, 1, 2))
for b in range(X.shape[3]):
  X[:, :, :, b] = (X[:, :, :, b] - means[b]) / stds[b]

# Randomize
idxs = list(range(y.shape[0]))
random.shuffle(idxs)

X = X[idxs]
y = y[idxs]

# Save data
out_file = out_dir + "/dataset.npz"
np.savez(out_file, X = X, y = y)
print("Saved data to: {}".format(out_file))
print("  'X' : {}".format(X.shape))
print("  'y' : {}".format(y.shape))
