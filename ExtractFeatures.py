import json
import numpy as np
from datetime import date, datetime, timedelta
import sys, getopt
import os
import librosa
import mysql.connector

# ExtractFeatures is a class that extract the wanted features
# form an audio file and store them in a SQL database.
# The configuration parameters are wrapped in a .json file
# Usage:
# python3 ExtractFeatures -i setup.json
# If -i is omitted, setup.json is anticipated to be the name of the json file

# Configuration parameters
N_MFCC = 13     # Number of MFCC elements
N_CHROMA = 12   # Number of Chroma elements
ROLLOFF_PERCENT = 0.85

audio_folder = ""
database = "recordings"
# Window and step size is set in seconds
window_size_sec = 0.05
step_size_sec = 0.025
windowing = False
use_energy_feature = False
use_zcr_feature = False
use_energy_entropy_feature = False
use_spectral_centroid_feature = False
use_spectral_spread_feature = False
use_spectral_flux_feature = False
use_spectral_rolloff_feature = False
use_mfcc_feature = False
use_chroma_feature = False
use_spectral_flatness_feature = False




# return the name of the configuration file.
def get_json_filename():
    # Check if -i argument is included in the command line
    try:
        myOpts, args = getopt.getopt(sys.argv[1:], "i:")
    except getopt.GetoptError as e:
        print(str(e))
        print("Usage: %s -i json_file.json" % sys.argv[0])
        sys.exit(2)
    file = ""
    if len(myOpts) > 0:
        for o, a in myOpts:
            if o == '-i':
                file = a
    if file != "":
        return file
    else:
        return "setup.json"


# Parse the .json configuration file
def get_config_params(jsonFileName):
    global audio_folder
    global database
    global window_size_sec
    global step_size_sec
    global windowing
    global use_energy_feature
    global use_zcr_feature
    global use_energy_entropy_feature
    global use_spectral_centroid_feature
    global use_spectral_spread_feature
    global use_spectral_flux_feature
    global use_spectral_rolloff_feature
    global use_mfcc_feature
    global use_chroma_feature
    global use_spectral_flatness_feature

    with open(jsonFileName) as jsonFile:
        configParam = json.load(jsonFile)
    # Update  configuration parameters
    for a in configParam:
        if a == "AudioFolder":
            audio_folder = configParam["AudioFolder"]
        if a == "Database":
            database = configParam["Database"]
        if a == "WindowSize":
            window_size_sec = configParam["WindowSize"]
        if a == "StepSize":
            step_size_sec = configParam["StepSize"]
        if a == "Windowing":
            if (configParam["Windowing"] == "Yes"):
                windowing = True
        if a == "Energy":
            if (configParam["Energy"] == "Yes"):
                use_energy_feature = True
        if a == "ZCR":
            if (configParam["ZCR"] == "Yes"):
                use_zcr_feature = True
        if a == "EnergyEntropy":
            if (configParam["EnergyEntropy"] == "Yes"):
                use_energy_entropy_feature = True
        if a == "SpectralCentroid":
            if (configParam["SpectralCentroid"] == "Yes"):
                use_spectral_centroid_feature = True
        if a == "SpectralSpread":
            if (configParam["SpectralSpread"] == "Yes"):
                use_spectral_spread_feature = True
        if a == "SpectralFlux":
            if (configParam["SpectralFlux"] == "Yes"):
                use_spectral_flux_feature = True
        if a == "SpectralRolloff":
            if (configParam["SpectralRolloff"] == "Yes"):
                use_spectral_rolloff_feature = True
        if a == "MFCC":
            if (configParam["MFCC"] == "Yes"):
                use_mfcc_feature = True
        if a == "Chroma":
            if (configParam["Chroma"] == "Yes"):
                use_chroma_feature = True
        if a == "SpectralFlatness":
            if (configParam["SpectralFlatness"] == "Yes"):
                use_spectral_flatness_feature = True

# Return a list of audio files
def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# Get the configuration parameters and list of audio files
json_file = get_json_filename()
print(json_file)
get_config_params(json_file)
audio_files = path_to_audiofiles(audio_folder)

# Loop through all audiofiles and calculate the features
for audio_file in audio_files:
    # Calculate features
    audio_samples, sample_rate = librosa.load(audio_file)
    window_size = int(window_size_sec * sample_rate)
    step_size = int(step_size_sec * sample_rate)
    print("Window_size: ", window_size, "Step_size: ", step_size)
    no_of_features = int((audio_samples.shape[0]-window_size)/step_size)
    # dt = time between each feature vector
    dt = step_size/sample_rate
    print("Extracting features from ", audio_file, "# samples: ", audio_samples.shape, " sr: ", sample_rate, " dt: ", dt, "# features: ", no_of_features)

    #---------------------
    # Extract selected features
    #---------------------
    if use_mfcc_feature:
        mfcc = librosa.feature.mfcc(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', n_mfcc=N_MFCC)
        print("MFCC Shape: ", mfcc.shape)
    if use_zcr_feature:
        zcr = librosa.feature.zero_crossing_rate(y=audio_samples, frame_length=window_size, hop_length=step_size)
    if use_spectral_flatness_feature:
        flatness = librosa.feature.spectral_flatness(y=audio_samples, hop_length=step_size, window='hann')
    if use_spectral_centroid_feature:
        centroid = librosa.feature.spectral_centroid(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann')
    if use_chroma_feature:
        chroma = librosa.feature.chroma_stft(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', n_chroma=N_CHROMA)
    if use_spectral_rolloff_feature:
        rolloff = librosa.feature.spectral_rolloff(y=audio_samples, sr=sample_rate, hop_length=step_size, window='hann', roll_percent=ROLLOFF_PERCENT)

    #---------------------
    # Save the features in the database
    #---------------------
    print(mysql.connector.__version__)
    cnx = mysql.connector.connect(user='parallels', password='ubuntuerbra', host='localhost', database=database)
    cursor = cnx.cursor()
    try:
        cursor.execute("USE {}".format(database))
    except mysql.connector.Error as err:
        print("Database {} does not exists.".format(database))
        print(err)
        exit(1)
    print("Database {} is selected".format(database))
    # Create the Recording table for this file
    date = str(datetime.now())
    date = date.split(".")[0]
    add_record_row = ("insert into Recording"
                        "(date, sound_file) "
                        "values (%s, %s)")
    try:
        cursor.execute(add_record_row, (date, audio_file))
    except mysql.connector.Error as err:
        print(err)
        exit(1)
    record_id = cursor.lastrowid
    # Create statistics tables for this file
    add_statistics_row = ("insert into Sound_statistics (parent, energy) values (%s, %s)")
    cursor.execute(add_statistics_row, (0, 0))
    std_id = cursor.lastrowid
    cursor.execute(add_statistics_row, (0, 0))
    mean_id = cursor.lastrowid
    cursor.execute(add_statistics_row, (0, 0))
    min_id = cursor.lastrowid
    cursor.execute(add_statistics_row, (0, 0))
    max_id = cursor.lastrowid

    # Create the Sound table for this file
    add_sound_row = ("insert into Sound"
                        "(sample_rate, window_size, step_size, std, mean, min, max) "
                        "values (%s, %s, %s, %s, %s, %s, %s)")
    try:
        cursor.execute(add_sound_row, (sample_rate, window_size, step_size, std_id, mean_id, min_id, max_id))
    except mysql.connector.Error as err:
        print(err)
        exit(1)
    sound_id = cursor.lastrowid
    # Update sound_id in record row
    print("Update the sound feature pointer in record")
    update_sound_id = ("update Recording set sound = %s where Recording.id = %s")
    try:
        cursor.execute(update_sound_id, (sound_id, record_id))
    except mysql.connector.Error as err:
        print(err)
        exit(1)

    print("Entering data for recording with id = ", record_id)
    # Create the firs row in the feature table and update the feature pointer in the sound table
    add_feature_row = ("insert into Sound_features set time_ms = %s, parent = %s")
    try:
        cursor.execute(add_feature_row, (0.0, sound_id))
    except mysql.connector.Error as err:
        print(err)
        exit(1)
    feature_id = cursor.lastrowid
    print("Entering feature pointer ")

    update_sound_id = ("update Sound set features = %s where Sound.id = %s")
    try:
        cursor.execute(update_sound_id, (feature_id, sound_id))
    except mysql.connector.Error as err:
        print(err)
        exit(1)
    print("Start entering feature data")
    loop=1
    for row in range(no_of_features):
        if row != 0:
            loop+=1
            # Create the next row in the feature table and update the next and prev pointers
            add_feature_row = ("insert into Sound_features set time_ms = %s, prev = %s, parent = %s")
            try:
                cursor.execute(add_feature_row, (dt*row, feature_id, sound_id))
            except mysql.connector.Error as err:
                print(err)
                exit(1)
            last_feature_id = feature_id
            feature_id = cursor.lastrowid
            add_feature_row = ("update Sound_features set next = %s where Sound_features.id = %s")
            try:
                cursor.execute(add_feature_row, (feature_id, last_feature_id))
            except mysql.connector.Error as err:
                print(err)
                exit(1)

        if use_mfcc_feature:
            # Save all MFCC elements
            for j in range(N_MFCC):
                # Replace one element in a record
                update_value = ("update Sound_features set mfcc%s = %s where Sound_features.id = %s")
                cursor.execute(update_value, (j, str(mfcc[j][row]), feature_id))
        if use_zcr_feature:
            update_value = ("update Sound_features set zcr = %s where Sound_features.id = %s")
            cursor.execute(update_value, (str(zcr[0][row]), feature_id))
        if use_spectral_flatness_feature:
            update_value = ("update Sound_features set spectral_flatness = %s where Sound_features.id = %s")
            cursor.execute(update_value, (str(flatness[0][row]), feature_id))
        if use_spectral_centroid_feature:
            update_value = ("update Sound_features set spectral_centroid = %s where Sound_features.id = %s")
            cursor.execute(update_value, (str(centroid[0][row]), feature_id))
        if use_chroma_feature:
            # Save all MFCC elements
            for j in range(N_CHROMA):
                # Replace one element in a record
                update_value = ("update Sound_features set chroma%s = %s where Sound_features.id = %s")
                cursor.execute(update_value, (j, str(chroma[j][row]), feature_id))
        if use_spectral_rolloff_feature:
            update_value = ("update Sound_features set spectral_rolloff = %s where Sound_features.id = %s")
            cursor.execute(update_value, (str(rolloff[0][row]), feature_id))
    #---------------------
    # Save the statistics
    #---------------------
    def save_statistics(y, name):
        std = np.std(y, axis=1)
        mean = np.mean(y, axis=1)
        min = np.min(y, axis=1)
        max = np.max(y, axis=1)
        command = "update Sound_statistics set "+name+"= %s where Sound_statistics.id = %s"
        cursor.execute(command, (str(std[0]), std_id))
        cursor.execute(command, (str(mean[0]), mean_id))
        cursor.execute(command, (str(min[0]), min_id))
        cursor.execute(command, (str(max[0]), max_id))


    if use_mfcc_feature:
        # Save statistics for all MFCC elements
        std = np.std(mfcc, axis=1)
        mean = np.mean(mfcc, axis=1)
        min = np.min(mfcc, axis=1)
        max = np.max(mfcc, axis=1)
        print("Std shape: ", std.shape)
        for j in range(N_MFCC):
            update_value = ("update Sound_statistics set mfcc%s = %s where Sound_statistics.id = %s")
            cursor.execute(update_value, (j, str(std[j]), std_id))
            cursor.execute(update_value, (j, str(mean[j]), mean_id))
            cursor.execute(update_value, (j, str(min[j]), min_id))
            cursor.execute(update_value, (j, str(max[j]), max_id))
    if use_chroma_feature:
        std = np.std(chroma, axis=1)
        mean = np.mean(chroma, axis=1)
        min = np.min(chroma, axis=1)
        max = np.max(chroma, axis=1)
        print("Std shape: ", std.shape)
        for j in range(N_CHROMA):
            update_value = ("update Sound_statistics set chroma%s = %s where Sound_statistics.id = %s")
            cursor.execute(update_value, (j, str(std[j]), std_id))
            cursor.execute(update_value, (j, str(mean[j]), mean_id))
            cursor.execute(update_value, (j, str(min[j]), min_id))
            cursor.execute(update_value, (j, str(max[j]), max_id))

    if use_zcr_feature:
        save_statistics(zcr, "zcr")
    if use_spectral_flatness_feature:
        save_statistics(flatness, "spectral_flatness")
    if use_spectral_centroid_feature:
        save_statistics(centroid, "spectral_centroid")
    if use_spectral_rolloff_feature:
        save_statistics(rolloff, "spectral_rolloff")


print("loop count: ", loop)
cursor.close()
cnx.commit()
cnx.close()

