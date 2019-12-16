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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from matplotlib import cm
from features import gen_training_data



json_file = "./cluster2D.json"
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
lt = int((cp["long_term"]-fcp["WindowSize"])/fcp["StepSize"])

print("Clustering and plotting")
data = np.reshape(data[0:int(int(data.shape[0]/lt)*lt),:], (int(data.shape[0]/lt), int(data.shape[1]*lt)))
# Make sure that each chunk of data belongs to the same label
temp = np.array([])
tmp_annotation = list()
for i in range(int(data.shape[0])):
    if str.split(annotation[i*lt], "-")[0] == str.split(annotation[i*lt+lt-1], "-")[0]:
        temp = np.append(temp, data[i, :])
        tmp_annotation.append(str.split(annotation[i*lt], ".")[0])
data = np.reshape(temp, (int(temp.shape[0]/(width*lt)), int(width*lt)))
annotation = tmp_annotation
print("Generated feature data and annotation shape:", data.shape, len(annotation))

print("Clustering data shape: ", data.shape)
embedding = SpectralEmbedding(n_components=cp["components"])
X_transformed = embedding.fit_transform(data)
print("Scatter shape: ", X_transformed.shape)
A = embedding.affinity_matrix_
print("Affinity matrix shape: ", A.shape)
print(A)

if cp["components"] == 3:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.subplot(1,2,1, projection='3d')
    ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2])
else:
    ax = plt.subplot()
    ax.scatter(X_transformed[:,0], X_transformed[:,1])
    for i, txt in enumerate(annotation):
        ax.annotate(annotation[i], (X_transformed[int(i),0], X_transformed[int(i),1]))
    plt.plot(X_transformed[:,0], X_transformed[:,1], 'bo')

plt.ylabel('some numbers')
plt.show()