from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import json
import sys, getopt
from features import gen_training_data
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
import pickle
import os.path
from os import path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


from sklearn.metrics import accuracy_score




model_file = open("model_summary.txt", "w")

def myprint(s):
    global model_file
    print(s, file=model_file)


# Get the parameters (both autoencoder and feature parameters) from json files
json_file = "./autoencoder_2D.json"
json_features_file = "./features.json"

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

# Generate training and initial annotation data sets
data, annotation = gen_training_data(json_features_file)
width = data.shape[1]
lt = int(cp["long_term"]/fcp["StepSize"])
data = np.reshape(data[0:int(int(data.shape[0]/lt)*lt),:], (int(data.shape[0]/lt), int(data.shape[1]*lt)))
print("Initial feature data set shape:", data.shape)

# Make sure that each chunk of data belongs to the same file
temp = np.array([])
tmp_annotation = list()
# The initial annotation is the filename without extension and directory path
# The label is normally the first characters of the filename ending with "_" (e.g. cr_2020-12-20). In the
# case of the Vowel files, a preceding number is added to the start of the file name (e.g. 9-i_h)
for i in range(int(data.shape[0])):
    if annotation[i*lt] == annotation[i*lt+lt-1]:
        temp = np.append(temp, data[i, :])
        tmp_annotation.append(str.split(str.split(annotation[i*lt], "_")[0], "-")[-1])
data = np.reshape(temp, (int(temp.shape[0]/(width*lt)), int(lt), int(width)))
annotation = tmp_annotation
print("Final feature data set shape:", data.shape)

image_size = data.shape
x_train = np.reshape(data, [-1, image_size[1], image_size[2], 1])

if not path.exists('autoencoder.json') or cp["force_train"]:

    # Network parameters
    input_shape = (image_size[1], image_size[2], 1)
    kernel_size = 3
    latent_dim = cp["code_size"]
    # Encoder/Decoder number of CNN layers and filters per layer
    #layer_filters = [128, 256]

    # Build the Autoencoder Model
    # First build the Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Stack of Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use MaxPooling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in cp["layer_filters"]:
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
    encoder.summary(print_fn=myprint)

    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # Stack of Transposed Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use UpSampling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in cp["layer_filters"][::-1]:
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
    decoder.summary(print_fn=myprint)

    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary(print_fn=myprint)
    model_file.close()

    autoencoder.compile(loss=cp["loss_function"], optimizer='adam', metrics=['accuracy'])

    # Train the autoencoder
    history = autoencoder.fit(x_train,
                    x_train,
                    validation_split=cp["validation_split"],
                    epochs=cp["epochs"],
                    batch_size=cp["batch_size"])

    # Save the models and weights
    # serialize model to JSON
    model_json = encoder.to_json()
    with open("encoder.json", "w") as json_file:
        json_file.write(model_json)
    model_json = autoencoder.to_json()
    with open("autoencoder.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encoder.save_weights("encoder.h5")
    autoencoder.save_weights("autoencoder.h5")
    print("Saved model to disk")
else:
    # Load the models
    json_file = open('encoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    encoder = model_from_json(loaded_model_json)
    # load weights into new model
    encoder.load_weights("encoder.h5")
    json_file = open('autoencoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder.load_weights("autoencoder.h5")
    print("Loaded model from disk")

# Produce the output of the code layer (latent_layer) for clustering

print("Generating data from the encoder")
x_pred = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
code = encoder.predict(x_train)
labels = np.unique(annotation)
clusters = labels.shape[0]
print("Number of unique labels: ", clusters)
clf = KMeans(n_clusters=clusters, random_state=42)
clf.fit(code)
# save the model to disk
filename = 'kmeans_model.sav'
pickle.dump(clf, open(filename, 'wb'))

print("K-Means labels:")
print(clf.labels_)

print("Clustering data shape: ", data.shape)
components = 3
embedding = SpectralEmbedding(n_components=components)
X_transformed = embedding.fit_transform(code)
print("Scatter shape: ", X_transformed.shape)
A = embedding.affinity_matrix_
print("Affinity matrix shape: ", A.shape)
print(A)

if components == 3:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.subplot(1, 2, 1, projection="3d")
    ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2])
    for i in range(code.shape[0]):
        ax.text(X_transformed[int(i), 0], X_transformed[int(i), 1], X_transformed[int(i), 2], '%s' % (str(clf.labels_[i])), size=10, zorder=1, color='k')
else:
    ax = plt.subplot()
    ax.scatter(X_transformed[:,0], X_transformed[:,1])
    for i in range(code.shape[0]):
        ax.annotate(annotation[i], (X_transformed[int(i),0], X_transformed[int(i),1]))
    plt.plot(X_transformed[:,0], X_transformed[:,1], 'bo')

plt.ylabel('some numbers')
plt.show()

# Find the corresponding label to the cluster number and save it as a numpy array
matrix = np.zeros([clusters, clusters])
for i, lbl in enumerate(labels):
    for j in range(clf.labels_.shape[0]):
        if annotation[j] == lbl:
            matrix[i, clf.labels_[j]] += 1
idx = np.argmax(matrix, axis=1)
sorted_labels = np.zeros((clusters), dtype="object")
for i in range(clusters):
    sorted_labels[i] = str(labels[int(np.where(idx==i)[0])])
np.save("sorted_labels.npy", sorted_labels)




# Show a plot of the input
image = x_train[0, 0:-1, 0:-1, 0]
image /= image.max()
fig, ax = plt.subplots()
extent = (0, x_train.shape[2], 0, x_train.shape[1])
ax.set_title("Input 0")
im = ax.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
fig.savefig("input1.png", format="png")
image = x_train[30, 0:-1, 0:-1, 0]
image /= image.max()
fig2, ax2 = plt.subplots()
extent = (0, x_train.shape[2], 0, x_train.shape[1])
ax2.set_title("Input 30")
im = ax2.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
fig2.savefig("input2.png", format="png")
# Show a plot of the output
pred = autoencoder.predict(x_train)
image = pred[0, 0:-1, 0:-1, 0]
image /= image.max()
fig3, ax3 = plt.subplots()
extent = (0, pred.shape[2], 0, pred.shape[1])
ax3.set_title("Output 0")
im = ax3.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
fig3.savefig("pred1.png", format="png")
image = pred[30, 0:-1, 0:-1, 0]
image /= image.max()
fig4, ax4 = plt.subplots()
extent = (0, pred.shape[2], 0, pred.shape[1])
ax4.set_title("Output 30")
im = ax4.imshow(image, cmap=plt.cm.hot, origin='upper', extent=extent)
fig4.savefig("pred2.png", format="png")
plt.show()

# Plot the result
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["Train loss", "Test loss"], loc='upper left')
plt.savefig("loss.png", format="png")
#plt.savefig(file_time+"figure.png", format="png")
plt.show()
