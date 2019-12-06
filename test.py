from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Reshape, UpSampling1D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.utils import plot_model


input_sig = Input(batch_shape=(1,128,1))
x = Conv1D(8,3, activation='relu', padding='same',dilation_rate=2)(input_sig)
x1 = MaxPooling1D(2)(x)
x2 = Conv1D(4,3, activation='relu', padding='same',dilation_rate=2)(x1)
x3 = MaxPooling1D(2)(x2)
x4 = AveragePooling1D()(x3)
flat = Flatten()(x4)
encoded = Dense(2)(flat)
d1 = Dense(64)(encoded)
d2 = Reshape((16,4))(d1)
d3 = Conv1D(4,1,strides=1, activation='relu', padding='same')(d2)
d4 = UpSampling1D(2)(d3)
d5 = Conv1D(8,1,strides=1, activation='relu', padding='same')(d4)
d6 = UpSampling1D(2)(d5)
d7 = UpSampling1D(2)(d6)
decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d7)
model= Model(input_sig, decoded)


plot_model(model, show_shapes=True, expand_nested=True, to_file='model.png')

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(model.summary())
