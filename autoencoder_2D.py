from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import json
import sys, getopt
from features import gen_training_data


json_file = "./autoencoder_2D.json"
json_features_file = "./features.json"

# Get the input json file
try:
    myOpts, args = getopt.getopt(sys.argv[1:], "i:")
except getopt.GetoptError as e:
    print(str(e))
    print("Usage: %s -i <cluster_config_file.json> -f <features_config_file.json>" % sys.argv[0])
    sys.exit(2)

for o, a in myOpts:
    if o == '-i':
        json_file = a
    if o == '-f':
        json_features_file = a
print(json_file)
with open(json_file) as file:
    cp = json.load(file)
with open(json_features_file) as file:
    fcp = json.load(file)

data, annotation = gen_training_data(json_features_file)
print("Generated feature data and annotation shape:", data.shape, len(annotation))
width = data.shape[1]
lt = int(cp["long_term"]/fcp["StepSize"])

#data[:, 13:19] = 0

print("Clustering and plotting")
data = np.reshape(data[0:int(int(data.shape[0]/lt)*lt),:], (int(data.shape[0]/lt), int(data.shape[1]*lt)))
# Make sure that each chunk of data belongs to the same label
temp = np.array([])
tmp_annotation = list()
for i in range(int(data.shape[0])):
    if str.split(annotation[i*lt], "-")[0] == str.split(annotation[i*lt+lt-1], "-")[0]:
        temp = np.append(temp, data[i, :])
        tmp_annotation.append(str.split(annotation[i*lt], ".")[0])
data = np.reshape(temp, (int(temp.shape[0]/(width*lt)), int(lt), int(width)))
annotation = tmp_annotation
print("Generated feature data and annotation shape:", data.shape, len(annotation))

image_size = data.shape
x_train = np.reshape(data, [-1, image_size[1], image_size[2], 1])

# Network parameters
input_shape = (image_size[1], image_size[2], 1)
kernel_size = 3
latent_dim = cp["code_size"]
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [128, 256]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Transposed Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UpSampling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation(cp["last_activation"], name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss=cp["loss_function"], optimizer='adam', metrics=['accuracy'])

# Train the autoencoder
history = autoencoder.fit(x_train,
                x_train,
                validation_split=cp["validation_split"],
                epochs=cp["epochs"],
                batch_size=cp["batch_size"])

# Show a plot of the input
image = x_train[0, 0:-1, 0:-1, 0]
image /= image.max()
fig, ax = plt.subplots()
extent = (0, x_train.shape[2], 0, x_train.shape[1])
ax.set_title("Input 0")
im = ax.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
image = x_train[30, 0:-1, 0:-1, 0]
image /= image.max()
fig2, ax2 = plt.subplots()
extent = (0, x_train.shape[2], 0, x_train.shape[1])
ax2.set_title("Input 30")
im = ax2.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
# Show a plot of the output
pred = autoencoder.predict(x_train)
image = pred[0, 0:-1, 0:-1, 0]
image /= image.max()
fig3, ax3 = plt.subplots()
extent = (0, pred.shape[2], 0, pred.shape[1])
ax3.set_title("Output 0")
im = ax3.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
image = pred[30, 0:-1, 0:-1, 0]
image /= image.max()
fig4, ax4 = plt.subplots()
extent = (0, pred.shape[2], 0, pred.shape[1])
ax4.set_title("Output 30")
im = ax4.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
plt.show()

# Plot the result
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["Train loss", "Test loss"], loc='upper left')
#plt.savefig("figure.png", format="png")
#plt.savefig(file_time+"figure.png", format="png")
plt.show()
