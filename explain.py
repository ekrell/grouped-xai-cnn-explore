# Run attribution methods
import numpy as np
from optparse import OptionParser
from scipy import ndimage
import tensorflow as tf
import keras
from skimage.segmentation import slic
import time

##############################
# XAI for 'grid' superpixels #
##############################

def calc_sums(attribs, patch_size):
  def apply_patch(in_image, out_image, top_left_x, top_left_y, patch_size):
    out_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = \
        np.sum(in_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size])
  rows, cols = attribs.shape
  img = np.zeros((rows, cols))
  for top_left_x in range(0, cols, patch_size):
    for top_left_y in range(0, rows, patch_size):
      apply_patch(attribs, img, top_left_x, top_left_y, patch_size)
  return img

def calc_occlusion(img, model, patch_size=2, class_idx=0, batch_size=512):
  def apply_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = 0
    return patched_image

  rows = img.shape[0]
  cols = img.shape[1]
  bands = img.shape[2]

  # Make first prediction
  original_prob = model.predict(img.reshape(1, rows, cols, bands), verbose=0)
  # Initialize storage
  sensitivity_map = np.zeros((rows, cols))
  patch_batch = np.zeros((batch_size, rows, cols, bands))
  all_predictions = []
  
  # Iterate the patch over the image and collect predictions
  b = 0
  for top_left_x in range(0, cols, patch_size):
    for top_left_y in range(0, rows, patch_size):
      # Mask out patch
      patched_image = np.array(apply_patch(img, top_left_x, top_left_y, patch_size))
      patch_batch[b] = patched_image
      b += 1
      # Predict batch
      if b == batch_size:
        predicted_classes = model.predict(patch_batch, verbose=0)
        all_predictions.append(predicted_classes[:, class_idx])
        b = 0

  # Predict with remaining batch
  predicted_classes = model.predict(patch_batch, verbose=0)
  all_predictions.append(predicted_classes[:, class_idx])
  # Combine predictions into single vector
  all_predictions = np.concatenate(all_predictions)
  # Use predictions to make occlusion map
  p_idx = 0
  for top_left_x in range(0, cols, patch_size):
    for top_left_y in range(0, rows, patch_size):
      confidence = all_predictions[p_idx]
      p_idx += 1
      diff = original_prob - confidence
      # Save confidence for this specific patched image in map
      sensitivity_map[
          top_left_y:top_left_y + patch_size,
          top_left_x:top_left_x + patch_size,
      ] = diff
  return sensitivity_map


###############################
# XAI for segment superpixels #
###############################

def calc_sums_segments(img, values, n_segments):
  sensitivity_map = np.zeros((img.shape[0], img.shape[1]))
  segments = slic(img, n_segments=n_segments, compactness=0.1)
  segment_idxs = segment_idxs = np.unique(segments.flatten())
  for segment in segments:
    for si in segment_idxs:
      sensitivity_map[segments == si] = np.sum(values[segments == si])
  return sensitivity_map


def calc_occlusion_segments(img, model, n_segments, class_idx=0, batch_size=512):
  segments = slic(img, n_segments=n_segments, compactness=0.1)
  sensitivity_map = np.zeros((img.shape[0], img.shape[1]))
  patch_batch = np.zeros((batch_size, rows, cols, bands))
  original_prob = predicted_classes = model.predict(np.array([img]), verbose=0)[0]

  all_predictions = []

  # Init patched image storage

  segment_idxs = np.unique(segments.flatten())
  b = 0

  # For each segment
  for si in segment_idxs:
    # Mask out patch
    patched_image = img.copy()
    patched_image[segments == si] = 0
    patch_batch[b] = patched_image
    b += 1

    # Predict batch
    if b == batch_size:
      predicted_classes = model.predict(patch_batch, verbose=0)
      all_predictions.append(predicted_classes[:, class_idx])
      b = 0

  # Predict with remaining batch
  predicted_classes = model.predict(patch_batch, verbose=0)
  all_predictions.append(predicted_classes[:, class_idx])
  # Combine predictions into single vector
  all_predictions = np.concatenate(all_predictions)
  # Use predictions to make occlusion map
  p_idx = 0
  for si in segment_idxs:
    confidence = all_predictions[p_idx]
    p_idx += 1
    diff = original_prob - confidence
    # Save confidence for this specific patched image in map
    sensitivity_map[segments == si
    ] = diff
  return sensitivity_map


# Options
parser = OptionParser()
parser.add_option("-d", "--data_file", default="tornado/data/dataset.npz",
                  help="Path to '.npz' with 'X' and 'y' variables.")
parser.add_option("-m", "--model_file", default="test-model.keras",
                  help="Path to load trained model.")
parser.add_option("-i", "--sample_idxs", default="0,1,2,3,4",
                  help="Comma-delimited list of samples to explain.")
parser.add_option("-s", "--scale_data", type=float, default=1.0,
                  help="Scale the input rows, cols by this.")
parser.add_option("-c", "--channels",
                  help="Comma-delimited list of bands to include.")
parser.add_option("-p", "--patch_sizes", default="1,2,4,6,8",
                  help="Comma-delimited list of superpixel patch sizes.")
parser.add_option("-o", "--out_file", default="test-attrs.npz",
                  help="Path to save attributions.")
parser.add_option(      "--y_name", default="y",
                  help="Name of 'y' variable in dataset.")
parser.add_option("-g", "--group_method", default="grid",
                  help="Name of pixel grouping method to use ('grid', 'slic').")
(options, args) = parser.parse_args()

data_file = options.data_file
scale_prop = options.scale_data
model_file = options.model_file
attrs_file = options.out_file
samples = np.array(options.sample_idxs.split(",")).astype(int)
patch_sizes = np.array(options.patch_sizes.split(",")).astype(int)
channel_idxs = np.array(options.channels.split(",")).astype(int) \
             if options.channels is not None else None
y_name = options.y_name

group_method = options.group_method

# Load data
data = np.load(data_file)
X = data["X"]
y = data[y_name]
# Select bands
if channel_idxs is not None:
  X = X[:, :, :, channel_idxs]
# Select samples
X = X[samples]
y = y[samples]
# Scale maps
order = 3
X = np.array([ndimage.zoom(x, (scale_prop, scale_prop, 1), order=order) for x in X])

# Load model
model = keras.models.load_model(model_file)
# Make predictions
ypred = model.predict(X, verbose=0)

samples, rows, cols, bands = X.shape
n_patch_sizes = patch_sizes.shape[0]

attrs = {
  "occlusion" : np.zeros((samples, n_patch_sizes, rows, cols)),
  "occlusion_sums" : np.zeros((samples, n_patch_sizes, rows, cols)),
}

# Run XAI methods
for sidx in range(samples):
  for pidx in range(n_patch_sizes):

    t0 = time.time()

    # Occlusion maps
    if group_method == "grid":
      attrs["occlusion"][sidx, pidx] = \
        calc_occlusion(X[sidx], model, patch_size=patch_sizes[pidx])
    else:
      attrs["occlusion"][sidx, pidx] = \
        calc_occlusion_segments(X[sidx], model, n_segments=patch_sizes[pidx])
    # Occlusion sums    
    if pidx == 0:
      if group_method == "grid":
        attrs["occlusion_sums"][sidx, pidx] = \
           calc_occlusion(X[sidx], model, patch_size=patch_sizes[pidx])
      else:
        attrs["occlusion_sums"][sidx, pidx] = \
          calc_occlusion_segments(X[sidx], model, n_segments=patch_sizes[pidx])
    else:
      if group_method == "grid":
        attrs["occlusion_sums"][sidx, pidx] = \
          calc_sums(attrs["occlusion_sums"][sidx, 0], patch_sizes[pidx])
      else:
        attrs["occlusion_sums"][sidx, pidx] = \
          calc_sums_segments(X[sidx], attrs["occlusion_sums"][sidx, 0], patch_sizes[pidx])

    print(sidx, "/", samples, "   ", time.time() - t0)
    t0 = time.time()


# Save
np.savez(attrs_file, 
         occlusion=attrs["occlusion"],
         occlusion_sums=attrs["occlusion_sums"],
         X=X, y=y, ypred=ypred,
         patch_sizes=patch_sizes,
)
