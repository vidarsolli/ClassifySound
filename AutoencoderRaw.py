from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import json
import time
import sys, getopt
import librosa
import os
import random


import numpy as np

labels = ['a_n', 'a_l','a_h', 'a_lhl','i_n', 'i_l','i_h', 'i_lhl', 'u_n', 'u_l','u_h', 'u_lhl',]

"""
AutodecoderRaw will train a 1D convolutional autoencoder with raw audio signals
and cluster the encoder vector to see if clusters contains sound with similar characteristics.
The parameters for the network are stored in a .json file

The audiofiles to be used are stored in a folder (folder to be stated in the .json file).

Usage:  python3 AutoencoderRaw -i <filenam>.json

"""

json_file = "./config.json"
# Return a list of audio files
def get_audiofiles(folder):
    list_of_audio = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (folder, file)
            list_of_audio.append(directory)
    return list_of_audio


# Get the input json file
try:
    myOpts, args = getopt.getopt(sys.argv[1:], "i:")
except getopt.GetoptError as e:
    print(str(e))
    print("Usage: %s -i <json_file>" % sys.argv[0])
    sys.exit(2)

for o, a in myOpts:
    if o == '-i':
        json_file = a

with open(json_file) as file:
    cp = json.load(file)

try:
    x_train = np.load("x_train.npy")
except:
    # Build training set
    audio_files = get_audiofiles(cp["audio_folder"])
    y_train = np.array([])
    x_train = np.array([])
    for file in audio_files:
        audio_samples, sample_rate = librosa.load(file)
        audio_samples=librosa.resample(audio_samples, sample_rate, cp["sample_rate"])
        window_size = int(cp["short_term"] * cp["sample_rate"])
        step_size = int(cp["step_size"] * cp["sample_rate"])
        print("Window_size: ", window_size, "Step_size: ", step_size)
        no_of_samples = int((audio_samples.shape[0]-window_size)/step_size)-1
        # dt = time between each feature vector
        dt = step_size/sample_rate
        print("Extracting features from ", file, "# samples: ", audio_samples.shape, " sr: ", cp["sample_rate"], " dt: ", dt, "# features: ", no_of_samples)

        for i in range(no_of_samples):
            y = audio_samples[(i*step_size):(i*step_size+window_size)]
            x_train = np.append(x_train, y)
            label = str.split(file, '.')[1]
            label = str.split(label, "-")[1]
            y_train = np.append(y_train, labels.index(label))

    x_train = np.reshape(x_train, (int(len(x_train)/window_size), window_size, 1))
    x_train += 1
    x_train *= 0.4
    min = np.min(x_train)
    max = np.max(x_train)
    print(min, max)
    print("X_train shape: ", x_train.shape)
    np.save("x_train", x_train)
    print("Y_train shape: ", y_train.shape)
    np.save("y_train", y_train)

input = Input(shape=(int(cp["sample_rate"]*cp["short_term"]), 1))  # adapt this if using `channels_first` image data format

x = Conv1D(cp["filter1_size"], cp["kernal1_size"], activation=cp["activation"], strides=cp["strides"], data_format='channels_last', padding='same')(input)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(cp["filter2_size"], cp["kernal2_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(cp["filter3_size"], cp["kernal3_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv1D(cp["filter3_size"], cp["kernal3_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(cp["filter2_size"], cp["kernal2_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(cp["filter1_size"], cp["kernal1_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 3, activation='linear', padding='same')(x)

print(type(decoded))

autoencoder = Model(input, decoded)

plot_model(autoencoder, show_shapes=True, expand_nested=True, to_file='model.png')

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Start the TensorBoard server by: tensorboard --logdir=/tmp/autoencoder
# and navigate to: http://0.0.0.0:6006
# In case, add: callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]

print(x_train.shape)

earlystopper = EarlyStopping(monitor='loss', min_delta=0.0001, patience=2, verbose=1)

if cp["train"]:
    autoencoder.fit(x_train, x_train,
                epochs=cp["epochs"],
                batch_size=cp["batch_size"],
                verbose=2,
                validation_split=cp["validation_split"],
                callbacks=[earlystopper])

# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")
