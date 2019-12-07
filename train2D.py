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

json_file = "./train2D.json"
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
print(json_file)
with open(json_file) as file:
    cp = json.load(file)

samples = cp["long_term"]/cp["step_size"]
samples -= (samples*cp["step_size"]-cp["long_term"])/cp["step_size"]
height = int(samples)
print("No of feature vectors per set: ", height)

data = gen_training_data("features.json")
print("Generated feature data shape:", data.shape)
width = data.shape[1]

# Build the training set
training_samples = 0
x_train = np.array([])
for i in range(data.shape[0]- height):
    x_train = np.append(x_train, data[i:i+height,:])
print("Training data shape (before reshape): ", x_train.shape)

x_train = np.reshape(x_train, (int(x_train.shape[0]/(height * width)), height, width, 1))
y_train = np.reshape(x_train, (x_train.shape[0], width*height))
print("Training data shape: ", x_train.shape)

print(x_train.shape)
input = Input(shape=(height, width, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(cp["filter1_size"], cp["kernal1_size"], activation=cp["activation"], strides=cp["strides"], data_format='channels_last', padding='same')(input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(cp["filter2_size"], cp["kernal2_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(cp["filter3_size"], cp["kernal3_size"], activation=cp["activation"], strides=cp["strides"], padding='same')(x)
x = MaxPooling2D(2, padding='same')(x)
encoded = Flatten()(x)
x = Dense(cp["dense1_size"], activation=cp["activation"])(encoded)
decoded = Dense(height*width, activation='linear') (x)

print(type(decoded))

autoencoder = Model(input, decoded)

plot_model(autoencoder, show_shapes=True, expand_nested=True, to_file='model.png')
print("Compiling model")
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(autoencoder.summary())

# Start the TensorBoard server by: tensorboard --logdir=/tmp/autoencoder
# and navigate to: http://0.0.0.0:6006
# In case, add: callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]

print(x_train.shape)

earlystopper = EarlyStopping(monitor='loss', min_delta=0.00001, patience=2, verbose=1)

if cp["train"]:
    history = autoencoder.fit(x_train, y_train,
                epochs=cp["epochs"],
                batch_size=cp["batch_size"],
                verbose=1,
                validation_split=cp["validation_split"])
#                callbacks=[earlystopper])

# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")

# Plot the result
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()