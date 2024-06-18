# Train simple CNN
import numpy as np
from scipy import ndimage
from optparse import OptionParser
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, \
     MaxPooling2D, Flatten, LeakyReLU, Dropout
from keras.layers import SpatialDropout2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K

# Options
parser = OptionParser()
parser.add_option("-d", "--data_file", default="tornado/data/dataset.npz",
                  help="Path to '.npz' with 'X' and 'y' variables.")
parser.add_option("-m", "--model_file", default="test-model.keras",
                  help="Path to save trained model.")
parser.add_option("-n", "--num_samples", type=int, default=10000,
                  help="Train with first N samples.")
parser.add_option("-s", "--scale_data", type=float, default=1.0,
                  help="Scale the input rows, cols by this.")
parser.add_option("-c", "--channels",
                  help="Comma-delimited list of bands to include.")
parser.add_option("-f", "--num_filters", type=int, default=8,
                  help="Number of convolutional filters")
parser.add_option("-w", "--filter_width", type=int, default=4,
                  help="Size of convolutional filters.")
parser.add_option("-l", "--learning_rate", type=float, default=0.001,
                  help="Learning rate.")
parser.add_option("-b", "--batch_size", type=int, default=512,
                  help="Batch size.")
parser.add_option("-e", "--epochs", type=int, default=50,
                  help="Number of epochs.")
(options, args) = parser.parse_args()

data_file = options.data_file
model_file = options.model_file
samples = np.array(range(options.num_samples))
scale_prop = options.scale_data
channel_idxs = np.array(options.channels.split(",")).astype(int) \
             if options.channels is not None else None
num_conv_filters = options.num_filters
filter_width = options.filter_width
learning_rate = options.learning_rate
batch_size = options.batch_size
epochs = options.epochs

conv_activation = "relu"

# Load data
data = np.load(data_file)
X = data["X"]
y = data["y"]
# Select bands
if channel_idxs is not None:
  X = X[:, :, :, channel_idxs]
# Select samples
X = X[samples]
y = y[samples]
# Scale maps 
order = 3 
X = np.array([ndimage.zoom(x, (scale_prop, scale_prop, 1), order=order) for x in X])

# Create model
# Input data in shape (instance, y, x, variable)
conv_net_in = Input(shape=X.shape[1:])
# First 2D convolution Layer
conv_net = Conv2D(num_conv_filters, (filter_width, filter_width), padding="same")(conv_net_in)
conv_net = Activation(conv_activation)(conv_net)
# Average pooling takes the mean in a 2x2 neighborhood to reduce the image size
conv_net = AveragePooling2D(pool_size=(2,2))(conv_net)
# Second set of convolution and pooling layers
conv_net = Conv2D(num_conv_filters * 2, (filter_width, filter_width), padding="same")(conv_net)
conv_net = Activation(conv_activation)(conv_net)
conv_net = AveragePooling2D(pool_size=(2,2))(conv_net)
# Third set of convolution and pooling layers
conv_net = Conv2D(num_conv_filters * 4, (filter_width, filter_width), padding="same")(conv_net)
conv_net = Activation(conv_activation)(conv_net)
conv_net = AveragePooling2D(pool_size=(2,2))(conv_net)
# Flatten the last convolutional layer into a long feature vector
conv_net = Flatten()(conv_net)
# Dense output layer, equivalent to a logistic regression on the last layer
conv_net = Dense(1)(conv_net)
conv_net = Activation("sigmoid")(conv_net)
model = Model(conv_net_in, conv_net)
# Use the Adam optimizer with default parameters
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(opt, "binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X, y, batch_size=batch_size, epochs=epochs)

# Save model
model.save(model_file)
print("Saved model: {}".format(model_file))
