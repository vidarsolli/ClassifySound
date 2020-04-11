import json
import numpy as np
from datetime import date, datetime, timedelta
import sys, getopt
import os
import librosa
import librosa.display

# features.py is a group of routines extracting features from audio files
# Configuration parameters are supplied via a .json file

# Configuration parameters
N_MFCC = 20     # Number of MFCC elements
N_CHROMA = 12   # Number of Chroma elements
N_MEL = 16      # Number of mel spectrum elements
ROLLOFF_PERCENT = 0.85


# Return a list of audio files
def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# Loop through all audiofiles and calculate the features
def gen_training_data(config_file):
    print("Generating training set based on:", config_file)
    with open(config_file) as json_file:
        cp = json.load(json_file)
    audio_files = path_to_audiofiles(cp["AudioFolder"])
    training_data = np.array([])
    annotation = list()
    for audio_file in audio_files:
        # Read the audio file and resample to wanted sample rate. If the file contains 2 channels,
        # select audio_samples[0]
        audio_samples, sample_rate = librosa.load(audio_file)
        audio_samples = librosa.resample(audio_samples, sample_rate, cp["SampleRate"])
        sample_rate = cp["SampleRate"]
        window_size = int(cp["WindowSize"] * sample_rate)
        step_size = int(cp["StepSize"] * sample_rate)
        audio_samples = librosa.effects.trim(audio_samples, top_db=20, frame_length=window_size, hop_length=step_size)

        audio_samples = np.array(audio_samples[0])

        # print("Window_size: ", window_size, "Step_size: ", step_size)
        no_of_samples = int((audio_samples.shape[0]-window_size)/step_size)
        # dt = time between each sample
        if no_of_samples > 0:
            dt = step_size/sample_rate
            print("Extracting features from ", audio_file, "# samples: ", audio_samples.shape, " sr: ", sample_rate, " dt: ", dt, "# samples: ", no_of_samples)

            #---------------------
            # Extract selected features
            #---------------------
            feature_vector = np.zeros((50, no_of_samples))
            vector_idx = 0
            if cp["MFCC"]:
                feature_vector[vector_idx:N_MFCC, :] = librosa.feature.mfcc(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', n_mfcc=N_MFCC)[:,0:no_of_samples]
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
                D = librosa.feature.melspectrogram(y=audio_samples, sr=sample_rate, hop_length=step_size, win_length=window_size, window='hann', center=True, n_mels=N_MEL)[:,0:no_of_samples]
                feature_vector[vector_idx:vector_idx+N_MEL, :] = librosa.amplitude_to_db(D, ref=np.max)
                vector_idx += N_MEL

            feature_vector = feature_vector.T
            #training_data[i * 4 + j, :, 0:13] = mfcc.T[step * j:timeseries_length + step * j, :]
            training_data = np.append(training_data, feature_vector[:,0:vector_idx])
            # Annotate the data with the filename
            label = str.split(str.split(audio_file, ".")[-2], "/")[-1]
            print(label)
            for i in range(no_of_samples):
                annotation.append(label)
    print(annotation)

    # Normailize the data
    training_data = np.reshape(training_data, (int(training_data.shape[0]/vector_idx), vector_idx))
    #print(training_data.shape)
    #print(training_data[2, :])
    std = np.std(training_data, axis=0)
    mean = np.mean(training_data, axis=0)
    min = np.min(training_data, axis=0)
    max = np.max(training_data, axis=0)
    #print(min)
    #print(max)
    if cp["Normalization"] == "MinMax":
        training_data = (training_data[:,]-min)/(max-min)
    if cp["Normalization"] == "Std":
        training_data = (training_data[:,]-mean)/std
    #print(training_data[2, :])
    np.save("features", training_data)
    np.save("annotation", annotation)

    return training_data, annotation
