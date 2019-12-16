from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import json
import sys, getopt
import matplotlib.pyplot as plt
from features import gen_training_data



import numpy as np

"""
AutodecoderRaw will train a 2D convolutional autoencoder with raw audio signals
and cluster the encoder vector to see if clusters contains sound with similar characteristics.
The parameters for the network are stored in a .json file

The audiofiles to be used are stored in a folder (folder to be stated in the .json file).

Usage:  python3 AutoencoderRaw -i <filenam>.json

"""

config_file = "./train2D.json"
features_file = "./features.json"

# Get the configuration json file
try:
    myOpts, args = getopt.getopt(sys.argv[1:], "i:")
except getopt.GetoptError as e:
    print(str(e))
    print("Usage: %s -i <config_file> -f <feature_file>" % sys.argv[0])
    sys.exit(2)

for o, a in myOpts:
    if o == '-i':
        config_file = a
    if o == '-f':
        features_file = a

with open(config_file) as file:
    cp = json.load(file)
with open(features_file) as file:
    fcp = json.load(file)

# Calculate number of feature vectors needed to cover the long-term period
samples = cp["long_term"]/fcp["StepSize"]
samples -= (samples*fcp["StepSize"]-cp["long_term"])/fcp["StepSize"]
height = int(samples)
print("No of feature vectors per set: ", height)

# Generate the data and annotation
data, annotation = gen_training_data("features.json")
print("Training data and annotation shape:", data.shape, len(annotation))
width = data.shape[1]

# Make sure that each chunk of data belongs to the same label, and
# that the annotation is correct
data = np.reshape(data[0:int(int(data.shape[0]/height)*height),:], (int(data.shape[0]/height), int(data.shape[1]*height)))
print("Data shape: ", data.shape)
temp_data = np.array([])
tmp_annotation = list()
for i in range(int(data.shape[0])):
    if annotation[i*height] == annotation[i*height+(height-1)]:
        temp_data = np.append(temp_data, data[i, :])
        tmp_annotation.append(str.split(annotation[i*height], ".")[0])
print("temp shape: ", data.shape)
data = temp_data
annotation = tmp_annotation
print("Data shape: ", data.shape)


#print("Training data shape (before reshape): ", x_train.shape)
# Reshape the data to match the network
x_train = np.reshape(data, (int(data.shape[0]/(height * width)), height, width, 1))
y_train = np.reshape(x_train, (x_train.shape[0], width*height))
print("Training data and test data shape: ", x_train.shape, y_train.shape)

# Build the autoencoder network model
input = Input(shape=(height, width, 1))  # adapt this if using `channels_first` image data format
x = Conv2D(cp["filter1_size"], cp["kernal1_size"], activation=cp["activation"], strides=cp["strides"], data_format='channels_last', padding='same')(input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(cp["filter2_size"], cp["kernal2_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(cp["filter3_size"], cp["kernal3_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = MaxPooling2D(2, padding='same')(x)
encoded = Flatten()(x)
x = Dense(cp["dense1_size"], activation=cp["activation"])(encoded)
x = Dense(cp["dense2_size"], activation=cp["activation"])(x)
decoded = Dense(height*width, activation='linear') (x)
autoencoder = Model(input, decoded)

# Save the model graph
plot_model(autoencoder, show_shapes=True, expand_nested=True, to_file='2Dmodel.png')

print("Compiling model")
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(autoencoder.summary())

# Start the TensorBoard server by: tensorboard --logdir=/tmp/autoencoder
# and navigate to: http://0.0.0.0:6006
# In case, add: callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]

earlystopper = EarlyStopping(monitor='loss', min_delta=0.00001, patience=2, verbose=1)

# Train the model
if cp["train"]:
    history = autoencoder.fit(x_train, y_train,
                epochs=cp["epochs"],
                batch_size=cp["batch_size"],
                verbose=2,
                validation_split=cp["validation_split"])
#                callbacks=[earlystopper])

# serialize model to JSON
model_json = autoencoder.to_json()
with open("2Dmodel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("2Dmodel.h5")
print("Saved model to disk")

# Plot accuracy training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss & Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Accuracy', 'Test'], loc='upper left')
plt.show()