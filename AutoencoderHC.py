from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import json
import sys, getopt
import matplotlib.pyplot as plt


json_file = "HCConfig.json"



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

config_file = cp["train_dir"]+"/"+str.split(cp["train_file"], '.')[0]+".json"
config_file = config_file.replace("training","config")
print(config_file)

with open(config_file) as file:
    feature_config = json.load(file)

feature_count = len(feature_config["Features"])
print(feature_count)


x_train = np.load(cp["train_dir"]+"/"+cp["train_file"])
print(x_train.shape)
std = np.std(x_train, 0)
min = np.min(x_train, 0)
max = np.max(x_train, 0)
print(std, min, max)

# this is the size of our encoded representations
encoding_dim = 2  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(feature_count,))

encoded = Dense(50, activation='relu')(input_img)
encoded = Dense(25, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(25, activation='relu')(encoded)
decoded = Dense(50, activation='relu')(decoded)
decoded = Dense(feature_count, activation='linear')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
plot_model(autoencoder, show_shapes=True, expand_nested=True, to_file='hcmodel.png')
print(autoencoder.summary())


print(x_train.shape)
# x_train = np.reshape(x_train, (len(x_train), feature_count, 1))

print(x_train[0,])
history = autoencoder.fit(x_train, x_train,
                epochs=500,
                batch_size=128,
                verbose = 1,
                shuffle=False,
                validation_split=cp["validation_split"])

# Plot the result
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test', 'Accuracy'], loc='upper left')
plt.show()