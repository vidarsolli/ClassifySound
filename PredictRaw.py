from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import json
import time
import sys, getopt
import librosa
import os
import random
import numpy as np
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt



labels = ['a_n', 'a_l','a_h', 'a_lhl','i_n', 'i_l','i_h', 'i_lhl', 'u_n', 'u_l','u_h', 'u_lhl',]


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
audio_files = get_audiofiles(cp["test_folder"])
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

    #for i in range(no_of_samples):
    for i in range(no_of_samples - 20,no_of_samples - 19):
        y = audio_samples[(i*step_size):(i*step_size+window_size)]
        x_train = np.append(x_train, y)
        label = str.split(file, '.')[1]
        label = str.split(label, "-")[1]
        #y_train = np.append(y_train, labels.index(label))
        y_train = np.append(y_train, label)

x_train = np.reshape(x_train, (int(len(x_train)/window_size), window_size, 1))
x_train += 1
x_train *= 0.4

# Restore the model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

# Read the data set for clustering
x_cluster = np.array([])
layer_name = 'conv1d_4'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

# Reading the output of the encoder layer
print("Reading the encoder output")
for i in range(x_train.shape[0]):
    x_predict = x_train[i]
    x_predict = np.reshape(x_predict, (1,800,1))
    # x_predict = x_train[i:i+1][:][:]
    intermediate_output = intermediate_layer_model.predict(x_predict)
    x_cluster = np.append(x_cluster, intermediate_output[0,:,0])
    # model.predict(x_predict, batch_size=1)
    # layer_value = model.layers[3].output.eval(session= K.get_session())
    # print(type(intermediate_output), intermediate_output.shape)
    #x_cluster = np.append(x_cluster, model.layers[3].output)
print("Clustering and plotting")
print(x_cluster.shape)
x_cluster = np.reshape(x_cluster, (x_train.shape[0], intermediate_output.shape[1]))
print(x_cluster.shape)
embedding = SpectralEmbedding(n_components=2)
X_transformed = embedding.fit_transform(x_cluster[:500])
print(X_transformed.shape)

fig, ax = plt.subplots()
ax.scatter(X_transformed[:,0], X_transformed[:,1])

for i, txt in enumerate(y_train[0:499]):
    ax.annotate(y_train[i], (X_transformed[i,0], X_transformed[i,1]))
#plt.plot(X_transformed[:,0], X_transformed[:,1], 'bo')
#plt.ylabel('some numbers')
plt.show()