from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import json
import time
import sys, getopt
import librosa
import os


import numpy as np

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

# Build training set

audio_files = get_audiofiles(cp["audio_folder"])
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

x_train = np.reshape(x_train, (int(len(x_train)/window_size), window_size, 1))
print(x_train.shape)
input = Input(shape=(int(cp["sample_rate"]*cp["short_term"]), 1))  # adapt this if using `channels_first` image data format

x = Conv1D(cp["filter1_size"], cp["kernal1_size"], activation=cp["activation"], data_format='channels_last', padding='same')(input)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(cp["filter2_size"], cp["kernal2_size"], activation=cp["activation"], padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(cp["filter3_size"], cp["kernal3_size"], activation=cp["activation"], padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv1D(cp["filter3_size"], cp["kernal3_size"], activation=cp["activation"], padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(cp["filter2_size"], cp["kernal2_size"], activation=cp["activation"], padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(cp["filter1_size"], cp["kernal1_size"], activation=cp["activation"], padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

print(type(decoded))

autoencoder = Model(input, decoded)

plot_model(autoencoder, show_shapes=True, expand_nested=True, to_file='model.png')

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Start the TensorBoard server by: tensorboard --logdir=/tmp/autoencoder
# and navigate to: http://0.0.0.0:6006
"""
autoencoder.fit(x_train, x_train,
                cp["epochs"],
                cp["batch_size"],
                shuffle=True,
                validation_data=(x_train, x_train),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
"""