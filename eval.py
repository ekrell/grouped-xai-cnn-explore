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
parser.add_option("-n", "--num_samples", type=int, default=10000,
                  help="Train with first N samples.")
parser.add_option("-s", "--scale_data", type=float, default=1.0,
                  help="Scale the input rows, cols by this.")
parser.add_option("-c", "--channels",
                  help="Comma-delimited list of bands to include.")
parser.add_option("-o", "--metrics_file",
                  help="Path to save metrics (.csv).")
parser.add_option("-i", "--indices_file",
                  help="Path to save outcome indices.")
parser.add_option(      "--y_name", default="y",
                  help="Name of 'y' variable in dataset.")
parser.add_option(      "--regression", default=False, action="store_true",
                  help="Evaluate a regression model.")
(options, args) = parser.parse_args()

data_file = options.data_file
model_file = options.model_file
scale_prop = options.scale_data
channel_idxs = np.array(options.channels.split(",")).astype(int) \
             if options.channels is not None else None
n_samples = options.num_samples

metrics_file = options.metrics_file
indices_file = options.indices_file

y_name = options.y_name
do_regression = options.regression

# Load data
data = np.load(data_file)
X = data["X"][:n_samples]
y = data[y_name][:n_samples].flatten()
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

# Regression evaluation
if do_regression:
  # Calc metrics
  mse = metrics.mean_squared_error(y, ypred)
  mae = metrics.mean_absolute_error(y, ypred)
  rmse = metrics.root_mean_squared_error(y, ypred)
  r2 = metrics.r2_score(y, ypred)

  # Get outcomes
  diffs = y - ypred
  adiffs = abs(diffs)
  maxdiff = np.max(adiffs)
  pdiffs = adiffs / maxdiff

  # Get outcomes
  best = np.where(pdiffs < 0.01)[0]
  n_best = len(best)
  close = np.where((pdiffs >= 0.01) & (pdiffs < 0.05))[0]
  n_close = len(close)
  middle = np.where((pdiffs >= 0.05) & (pdiffs < 0.1))[0]
  n_middle = len(middle)
  worst = np.where(pdiffs >= 0.1)[0]
  n_worst = len(worst)

  print("best: {}   close: {}   middle: {}   worst: {}".format(
        n_best, n_close, n_middle, n_worst))
  print("MSE: {:.4f}   RMSE: {:.4f}   MAE: {:.4f}   R2: {:.4f}".format(
        mse, rmse, mae, r2))

  # Write metrics
  with open(metrics_file, "w") as f:
    f.write("best,close,middle,worst,mse,rmse,mae,r2\n")
    f.write("{},{},{},{},{},{},{},{}\n".format(
      n_best, n_close, n_middle, n_worst, mse, rmse, mae, r2))

  # Write indices of each outcome
  with open(indices_file, 'w') as f:
    f.write("{}\t{}\n".format("best", ",".join([str(v) for v in best])))
    f.write("{}\t{}\n".format("close", ",".join([str(v) for v in close])))
    f.write("{}\t{}\n".format("middle", ",".join([str(v) for v in middle])))
    f.write("{}\t{}\n".format("worst", ",".join([str(v) for v in worst])))

# Classification evaluation
else:
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
