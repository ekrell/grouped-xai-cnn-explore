# Plot attributions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from optparse import OptionParser
import tensorflow as tf
import keras

def calc_mean_corrs(attrs):
  reps, maps, rows, cols = attrs.shape
  mean_corrs = np.zeros(maps)
  compares = list(itertools.combinations(range(reps), 2))
  for m in range(maps):
    for compare in compares:
      mean_corrs[m] += np.corrcoef(attrs[compare[0], m].flatten(),
                                   attrs[compare[1], m].flatten())[0,1]
    mean_corrs[m] = mean_corrs[m] / len(compares)
  return mean_corrs


def plot_attrs(attrs, valign=False, inverty=True, title="", meancorrs=None,
               top_labels=None, divides=None, y_labels=None, y_right_labels=None, metric_name=""):
  reps, maps, rows, cols = attrs.shape
  fig, axs = plt.subplots(reps, maps, squeeze=False, figsize=(maps*1.5, reps*1.5))
  fig.suptitle(title)

  attrs_ = attrs

  if divides is not None:
    for r in range(reps):
      for m in range(maps):
        attrs_[r, m] =  attrs_[r, m] / divides[m]

  if valign:
    vmin_ = np.nanmin(attrs)
    vmax_ = np.nanmax(attrs)
    vmax = max(abs(vmin_), abs(vmax_))
    vmin = -vmax

  for r in range(reps):
    for m in range(maps):
      data = attrs_[r, m]
      if not valign:
        vmin_ = np.nanmin(data)
        vmax_ = np.nanmax(data)
        vmax = max(abs(vmin_), abs(vmax_))
        vmin = -vmax
      axs[r, m].imshow(data, vmin=vmin, vmax=vmax, cmap="bwr")
      axs[r, m].set_xticks([])
      axs[r, m].set_yticks([])
      if inverty:
        axs[r, m].invert_yaxis()

  if meancorrs is not None:
    for m in range(maps):
      axs[-1, m].set_xlabel("{0:.4f}".format(meancorrs[m]))
      fig.supxlabel("Mean of pairwise correlation across column attributions")
  if top_labels is not None:
    for m in range(maps):
      axs[0, m].set_title("{}".format(top_labels[m]))
  if y_labels is not None:
    for r in range(reps):
      axs[r, 0].set_ylabel("model {0}\npred = {1:.3f}".format(r, y_labels[r]))
      fig.supylabel("Each row corresponds to a trained model")
  if y_right_labels is not None:
    for r in range(reps):    
      axr = axs[r,-1].twinx()
      axr.set_yticks([])
      axr.set_ylabel("{}\n= {:.3f}".format(metric_name, y_right_labels[r]))
      
  plt.tight_layout()


def make_html_table(files):
  html = """


    <center>
    <img border="2px solid #000" height="300" src="{}"></img>
    <img border="2px solid #000" height="300" src="{}"></img>

    <table>
    <tr>
      <td><img src="{}"></img></td>
      <td><img src="{}"></img></td>
    </tr>
    </table>
    </center>

  """.format(files[5], files[4], files[1], files[3])
  return html


# Options
parser = OptionParser()
parser.add_option("-f", "--attr_files", 
                  help="Semicolon-delimited list of attribution files.")
parser.add_option("-m", "--metric_files",
                  help="Semicolon-delimited list of metrics files.")
parser.add_option("-o", "--out_dir", 
                  help="Path to output directory.")
parser.add_option(      "--metric", default="balanced_accuracy",
                  help="Name of metric to plot.")
parser.add_option(      "--use_segments", default=False, action="store_true",
                  help="Specify that superpixels are clusters, not squares.")
(options, args) = parser.parse_args()
attrs_files = options.attr_files.split(";")
out_dir = options.out_dir
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
y_name = "y"
metric_name = options.metric
metric_name_str = metric_name
if len(metric_name_str) > 12:  # Abbreviate long name
  metric_name_str = metric_name_str[:11]

metric_files = options.metric_files.split(";") if options.metric_files is not None else None

use_segments = options.use_segments
use_segments = True

# Load attributions
occs = np.stack([np.load(attrs_file)["occlusion"] for attrs_file in attrs_files], axis=1)
sums = np.stack([np.load(attrs_file)["occlusion_sums"] for attrs_file in attrs_files], axis=1)

# Load data
X = np.load(attrs_files[0])["X"]
y = np.load(attrs_files[0])[y_name]
n_bands = X.shape[-1]

# Load metadata
patch_sizes = np.load(attrs_files[0])["patch_sizes"]

if use_segments == False:
  width = X.shape[2]
  patch_sizes = [p for p in patch_sizes if p < width]
  patch_labels = ["{}x{}".format(ps, ps) for ps in patch_sizes]
  occs = occs[:,:,:len(patch_sizes),...]
  sums = sums[:,:,:len(patch_sizes),...]
else:
  patch_labels = ["{}".format(ps) for ps in patch_sizes]

n_samples, n_reps, n_maps, n_rows, n_cols = occs.shape
samples = np.array(range(n_samples))

# Load predictions
preds = np.squeeze(np.stack([np.load(attrs_file)["ypred"] for attrs_file in attrs_files], axis=1))

# Load metrics
if metric_files is not None:
  dfMetrics = pd.concat([pd.read_csv(mf) for mf in metric_files])
  y_right_labels = dfMetrics[metric_name].values
else:
  y_right_labels = None

outfiles = [
  ("sample-" + str(si) + "_occs_n.png",
   "sample-" + str(si) + "_occs_v.png",
   "sample-" + str(si) + "_sums_n.png",
   "sample-" + str(si) + "_sums_v.png",
   "sample-" + str(si) + "_input.png",
   "sample-" + str(si) + "_input_img.png",
  ) for si in samples]


# Init HTML report
html = ""

for sample in samples:
  # Calculate correlations
  occs_mean_corrs = calc_mean_corrs(occs[sample])
  sums_mean_corrs = calc_mean_corrs(sums[sample])

  target = y[sample]

  # Plot sample (contour maps)
  fig, axs = plt.subplots(1, n_bands, squeeze=False, figsize=(4*n_bands, 4))
  for b in range(n_bands):
    axs[0, b].contour(X[sample, :, :, b])
    axs[0, b].axis("off")
    #axs[0, b].invert_yaxis()
  fig.suptitle("Sample {}  |  target = {}".format(sample, target))
  plt.tight_layout()
  plt.savefig(out_dir + outfiles[sample][4])

  # Plot sample (image)
  fig, axs = plt.subplots(1, n_bands, squeeze=False, figsize=(4*n_bands, 4))
  for b in range(n_bands):
    axs[0, b].imshow(X[sample, :, :, b], cmap="Greys")
    axs[0, b].axis("off")
    axs[0, b].invert_yaxis()
  fig.suptitle("Sample {}  |  target = {}".format(sample, target))
  plt.tight_layout()
  plt.savefig(out_dir + outfiles[sample][5])
 
  y_labels = preds[sample]
  if np.isscalar(y_labels):
    y_labels = np.atleast_1d(y_labels)

  # Plot attribution maps
  plot_attrs(occs[sample], top_labels=patch_labels, y_labels=y_labels,
      title="Sample {}  |  target: {}  |  Occlusion Maps".format(sample, target),
      meancorrs=occs_mean_corrs, y_right_labels=y_right_labels, metric_name=metric_name_str)
  plt.savefig(out_dir + outfiles[sample][0])
  plot_attrs(occs[sample], valign=True,  top_labels=patch_labels, y_labels=y_labels,
      title="Sample {}  |  target: {}  |  Occlusion Maps".format(sample, target), divides=patch_sizes,
      meancorrs=occs_mean_corrs, y_right_labels=y_right_labels, metric_name=metric_name_str)
  plt.savefig(out_dir + outfiles[sample][1])
  plot_attrs(sums[sample],  top_labels=patch_labels, y_labels=y_labels,
      title="Sample {}  |  target: {}  |  Occlusion Sums".format(sample, target),
      meancorrs=sums_mean_corrs, y_right_labels=y_right_labels, metric_name=metric_name_str)
  plt.savefig(out_dir + outfiles[sample][2])
  plot_attrs(sums[sample], valign=True, top_labels=patch_labels, y_labels=y_labels,
      title="Sample {}  |  target: {}  |  Occlusion Sums".format(sample, target), divides=patch_sizes,
      meancorrs=sums_mean_corrs, y_right_labels=y_right_labels, metric_name=metric_name_str)
  plt.savefig(out_dir + outfiles[sample][3])

  # Add plots to HTML report
  html += make_html_table(outfiles[sample])
  plt.close()

# Save HTML report
f = open(out_dir + "/report.html", 'w') 
f.write(html)
f.close()
print("Saved HTML report: {}".format(out_dir + "/report.html"))
