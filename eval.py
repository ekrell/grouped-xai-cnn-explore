# Evaluate model
import numpy as np
import pandas as pd
from optparse import OptionParser
from scipy import ndimage
import tensorflow as tf
import keras
from sklearn import metrics

# Options
parser = OptionParser()
parser.add_option("-d", "--data_file", default="tornado/data/dataset.npz",
                  help="Path to '.npz' with 'X' and 'y' variables.")
parser.add_option("-m", "--model_file", default="test-model.keras",
                  help="Path to load trained model.")
parser.add_option("-s", "--scale_data", type=float, default=1.0,
                  help="Scale the input rows, cols by this.")
parser.add_option("-c", "--channels",
                  help="Comma-delimited list of bands to include.")
parser.add_option("-o", "--metrics_file",
                  help="Path to save metrics (.csv).")
parser.add_option("-i", "--indices_file",
                  help="Path to save outcome indices.")
(options, args) = parser.parse_args()

data_file = options.data_file
model_file = options.model_file
scale_prop = options.scale_data
channel_idxs = np.array(options.channels.split(",")).astype(int) \
             if options.channels is not None else None

metrics_file = options.metrics_file
indices_file = options.indices_file

# Load data
data = np.load(data_file)
X = data["X"]
y = data["y"]
# Select bands
if channel_idxs is not None:
  X = X[:, :, :, channel_idxs]
# Scale maps
order = 3
X = np.array([ndimage.zoom(x, (scale_prop, scale_prop, 1), order=order) for x in X])
N, rows, cols, bands = X.shape
# Load model
model = keras.models.load_model(model_file)

# Make predictions
ypred = model.predict(X)[:, 0]
# Convert to binary decision
classes = (ypred > 0.5).astype("int32")

# Calc metrics
accuracy = metrics.accuracy_score(y, classes) 
balanced_accuracy = metrics.balanced_accuracy_score(y, classes) 
TN, FP, FN, TP = metrics.confusion_matrix(y, classes).ravel()

# Get outcomes
outcomes = pd.Series([" " for i in range(N)]).astype(str)
outcomes[np.logical_and(classes == 1, y == 1)] = "TP"
outcomes[np.logical_and(classes == 0, y == 0)] = "TN"
outcomes[np.logical_and(classes == 1, y == 0)] = "FP"
outcomes[np.logical_and(classes == 0, y == 1)] = "FN"

print("TP: {}   TN: {}   FP: {}   FN: {}".format(TP, TN, FP, FN))
print("Accuracy: {:.4f}   Balanced accuracy: {:.4f}".format(accuracy, balanced_accuracy))

# Write metrics 
with open(metrics_file, "w") as f:
  f.write("TP,TN,FP,FN,accuracy,balanced_accuracy\n")
  f.write("{},{},{},{},{:.4},{:.4}\n".format(TP, TN, FP, FN, accuracy, balanced_accuracy))

# Write indices of each outcome
with open(indices_file, 'w') as f:
  f.write("{}\t{}\n".format("TP", ",".join([str(v) for v in np.where(outcomes == "TP")[0]])))
  f.write("{}\t{}\n".format("TN", ",".join([str(v) for v in np.where(outcomes == "TN")[0]])))
  f.write("{}\t{}\n".format("FP", ",".join([str(v) for v in np.where(outcomes == "FP")[0]])))
  f.write("{}\t{}\n".format("FN", ",".join([str(v) for v in np.where(outcomes == "FN")[0]])))
