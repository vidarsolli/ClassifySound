from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import json
import sys, getopt

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




# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(feature_count,))

encoded = Dense(100, activation='relu')(input_img)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(feature_count, activation='linear')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print(x_train.shape)
# x_train = np.reshape(x_train, (len(x_train), feature_count, 1))

print(x_train[0,])
autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=32,
                verbose = 1,
                shuffle=True,
                validation_split=cp["validation_split"])

