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
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
print(x_train.shape, y_train.shape)
print(y_train)
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
    ax.annotate(labels[int(txt)], (X_transformed[i,0], X_transformed[i,1]))
#plt.plot(X_transformed[:,0], X_transformed[:,1], 'bo')
#plt.ylabel('some numbers')
plt.show()