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
from sklearn.decomposition import PCA
import os.path
import numpy as np
from joblib import dump, load


"""
AutodecoderRaw will train a 2D convolutional autoencoder with raw audio signals
and cluster the encoder vector to see if clusters contains sound with similar characteristics.
The parameters for the network are stored in a .json file

The audiofiles to be used are stored in a folder (folder to be stated in the .json file).

Usage:  python3 AutoencoderRaw -i <filenam>.json

"""

config_file = "./pca2D.json"
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

if os.path.isfile('features.npy'):
    print ("Reading feature file")
    data = np.load("features.npy")
    annotation = np.load("annotation.npy")
else:
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
x_train = np.reshape(data, (int(data.shape[0]/(height * width)), int(height*width)))
y_train = np.reshape(x_train, (x_train.shape[0], width*height))
print("Training data and test data shape: ", x_train.shape, y_train.shape)

pca_sound = PCA(n_components=cp["no_of_components"])
pca_components = pca_sound.fit_transform(x_train)
print(pca_components.shape)
print(pca_sound.get_params())

dump(pca_sound, "pca.model")

pca_model = load("pca.model")
pca_components = pca_model.transform(x_train)

unique_annotations = np.unique(annotation)
print(unique_annotations.shape)
print(unique_annotations)

# Plot the result
fig, ax = plt.subplots(1, len(cp["plot"])+1)
#ax1.figure()
#ax1.figure(figsize=(10,10))
#fig.xticks(fontsize=12)
#fig.yticks(fontsize=14)
#ax1.xlabel('Principal Component - 1',fontsize=20)
#ax1.ylabel('Principal Component - 2',fontsize=20)
#fig.title("Principal Component Analysis of Sound Dataset",fontsize=20)
ax[0].set_title("Principle components")
ax[0].scatter(pca_components[:, 0], pca_components[:, 1], c = "r", s = 50)
for i, txt in enumerate(annotation):
    ax[0].annotate(annotation[i], (pca_components[int(i), 0], pca_components[int(i), 1]))

# Plot the trace for one or more sound sequences
annotation = np.array(annotation)
for j in range(len(cp["plot"])):
    arr_index = np.where(annotation == cp["plot"][j])
    ax[j+1].set_title(cp["plot"][j])
    ax[j+1].set_xlim(ax[0].get_xlim())
    ax[j+1].set_ylim(ax[0].get_ylim())
    arr_index = np.array(arr_index)
    for i in range(arr_index.shape[1]-1):
        ax[j+1].plot(pca_components[arr_index[0,0], 0],pca_components[arr_index[0,0], 1], "ro")
        ax[j+1].plot(pca_components[arr_index[0,:], 0],pca_components[arr_index[0,:], 1], linestyle=":")

#plt.plot(X_transformed[:, 0], X_transformed[:, 1], 'bo')

plt.show()
#               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
#targets = ['Benign', 'Malignant']
#colors = ['r', 'g']
#for target, color in zip(targets,colors):
#    indicesToKeep = breast_dataset['label'] == target
#    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
#               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

#plt.legend(targets,prop={'size': 15})