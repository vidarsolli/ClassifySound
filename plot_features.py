import json
import numpy as np
from datetime import date, datetime, timedelta
import sys, getopt
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


# features.py is a group of routines for extracting features from audio files
# Configuration parameters are supplied via a .json file

# Configuration parameters
N_MFCC = 13     # Number of MFCC elements
N_CHROMA = 12   # Number of Chroma elements
N_MEL = 16      # Number of mel spectrum elements
ROLLOFF_PERCENT = 0.85

y, sr = librosa.load("./audio/out/L_misfornøyd2.wav")
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

D = librosa.feature.mfcc(y=y, sr=sr, hop_length=128, window='hann', n_mfcc=N_MFCC)
print(D.shape)
plt.subplot(5, 2, 4)
librosa.display.specshow(D, y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
plt.title('mfcc')

D = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=128, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0, n_mels=16)
plt.subplot(5, 2, 6)
D = librosa.amplitude_to_db(D, ref=np.max)
print(D.shape)
librosa.display.specshow(D, x_axis='time', y_axis='log')
#plt.colorbar(format='%+2.0f dB')
plt.title('mel spectrogram')

plt.subplots_adjust(hspace=1)

plt.show()