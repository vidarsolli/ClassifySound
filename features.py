import json
import numpy as np
from datetime import date, datetime, timedelta
import sys, getopt
import os
import librosa
import librosa.display

# features.py is a group of routines for extracting features from audio files
# Configuration parameters are supplied via a .json file

# Configuration parameters
N_MFCC = 13     # Number of MFCC elements
N_CHROMA = 12   # Number of Chroma elements
ROLLOFF_PERCENT = 0.85


# Return a list of audio files
def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# Loop through all audiofiles and calculate the features
def gen_training_data(config_file):
    with open(config_file) as json_file:
        cp = json.load(json_file)
    audio_files = path_to_audiofiles(cp["AudioFolder"])
    training_data = np.array([])
    for audio_file in audio_files:
        # Read the audio file and resample to wanted sample rate. If the file contains 2 channels,
        # select audio_samples[0]
        window_size = int(cp["WindowSize"] * sample_rate)
        step_size = int(cp["StepSize"] * sample_rate)
        audio_samples, sample_rate = librosa.load(audio_file)
        audio_samples = librosa.resample(audio_samples, sample_rate, cp["SampleRate"])
        audio_samples = librosa.effects.trim(audio_samples, top_db=20, frame_length=window_size, hop_length=step_size)
        sample_rate = cp["SampleRate"]

        # print("Window_size: ", window_size, "Step_size: ", step_size)
        no_of_samples = int((audio_samples.shape[0]-window_size)/step_size)
        # dt = time between each sample
        dt = step_size/sample_rate
        print("Extracting features from ", audio_file, "# samples: ", audio_samples.shape, " sr: ", sample_rate, " dt: ", dt, "# samples: ", no_of_samples)

        #---------------------
        # Extract selected features
        #---------------------
        feature_vector = np.zeros((40, no_of_samples))
        vector_idx = 0
        if cp["MFCC"]:
            feature_vector[0:N_MFCC, :] = librosa.feature.mfcc(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', n_mfcc=N_MFCC)[:,0:no_of_samples]
            vector_idx += N_MFCC
        if cp["ZCR"]:
            feature_vector[vector_idx:vector_idx+1, :] = librosa.feature.zero_crossing_rate(y=audio_samples, frame_length=window_size, hop_length=step_size)[:,0:no_of_samples]
            vector_idx += 1
        if cp["SpectralFlatness"]:
            feature_vector[vector_idx:vector_idx+1, :] = librosa.feature.spectral_flatness(y=audio_samples, hop_length=step_size, window='hann')[:,0:no_of_samples]
            vector_idx += 1
        if cp["SpectralCentroid"]:
            feature_vector[vector_idx:vector_idx+1, :] = librosa.feature.spectral_centroid(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann')[:,0:no_of_samples]
            vector_idx += 1
        if cp["Chroma"]:
            feature_vector[vector_idx:vector_idx+N_CHROMA, :] = librosa.feature.chroma_stft(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', n_chroma=N_CHROMA)[:,0:no_of_samples]
            vector_idx += N_CHROMA
        if cp["SpectralRolloff"]:
            feature_vector[vector_idx:vector_idx+1, :] = librosa.feature.spectral_rolloff(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', roll_percent=ROLLOFF_PERCENT)[:,0:no_of_samples]
            vector_idx += 1
        if cp["Mel"]:
            D = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=step_size, win_length=None, window='hann', center=True)
            feature_vector[vector_idx:vector_idx+1, :] = librosa.amplitude_to_db(D, ref=np.max)

        feature_vector = feature_vector.T
        #training_data[i * 4 + j, :, 0:13] = mfcc.T[step * j:timeseries_length + step * j, :]
        training_data = np.append(training_data, feature_vector[:,0:vector_idx])
    # Normailize the data
    training_data = np.reshape(training_data, (int(training_data.shape[0]/vector_idx), vector_idx))
    print(training_data.shape)
#    print(training_data[2, :])
    std = np.std(training_data, axis=0)
    mean = np.mean(training_data, axis=0)
    min = np.min(training_data, axis=0)
    max = np.max(training_data, axis=0)
    print(min)
    print(max)
    if cp["Normalization"] == "MinMax":
        training_data = (training_data[:,]-min)/(max-min)
#    print(training_data[2, :])

    return training_data

#training_set = gen_training_data("features.json")
#print(training_set.shape)

import matplotlib.pyplot as plt
y, sr = librosa.load("./audio/out/Stille2.wav")
print(sr)
print("Duration before:", librosa.get_duration(y))
y, index = librosa.effects.trim(y, top_db=20, frame_length= 512, hop_length = 128)
print("Duration after:", librosa.get_duration(y))

plt.subplot(6, 2, 1)
#x = librosa.core.amplitude_to_db(y, ref=np.max(y), amin=1e-05, top_db=80.0)
plt.plot( y, 'r')
plt.title('raw signal')


D = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=128, window='hann')
plt.subplot(6, 2, 3)
plt.plot(range(D.shape[1]), D[0,:], 'r')
plt.title('spectral centroid')

D = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=128)
plt.subplot(5, 2, 5)
plt.plot(range(D.shape[1]), D[0,:], 'r')
plt.title('zcr')

D = librosa.feature.spectral_flatness(y=y, hop_length=128, window='hann')
plt.subplot(5, 2, 7)
plt.plot(range(D.shape[1]), D[0,:], 'r')
plt.title('spectral flatness')

D = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=128, window='hann', roll_percent=ROLLOFF_PERCENT)
plt.subplot(5, 2, 9)
plt.plot(D[0,:], 'r')
plt.title('roll-off')

D = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=128, window='hann', n_chroma=N_CHROMA)
print(D.shape)
plt.subplot(5, 2, 2)
librosa.display.specshow(D, y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
plt.title('chroma spectrogram')

D = librosa.feature.mfcc(y=y, sr=sr, hop_length=64, window='hann', n_mfcc=N_MFCC)
print(D.shape)
plt.subplot(5, 2, 4)
librosa.display.specshow(D, y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
plt.title('mfcc')

D = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=128, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)
plt.subplot(5, 2, 6)
D = librosa.amplitude_to_db(D, ref=np.max)
D = np.mean(D[])
print(D.shape)
librosa.display.specshow(D, x_axis='time', y_axis='log')
#plt.colorbar(format='%+2.0f dB')
plt.title('mel spectrogram')

plt.subplots_adjust(hspace=1)

plt.show()