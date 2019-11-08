import json
import numpy as np
from datetime import date, datetime, timedelta
import sys, getopt
import os
import librosa
import mysql.connector

# GenerateDataSets parses the database and generates datasets
# according to a .json configuration fil

# Configuration parameters
database = "recordings"
selected_features = ["no"]
selected_records = ['no']

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
        return "DataSetConfig.json"

# Parse the .json configuration file
def get_config_params(jsonFileName):
    global database
    global selected_features
    global selected_records

    with open(jsonFileName) as jsonFile:
        configParam = json.load(jsonFile)
    # Update  configuration parameters
    for a in configParam:
        if a == "Database":
            database = configParam["Database"]
        if a == "Features":
            selected_features = configParam["Features"]
            print("Selected features: ", selected_features)
        if a == "Records":
            selected_records = configParam["Records"]
            print("Selected records: ", selected_records)

# Get the configuration parameters and list of audio files
json_file = get_json_filename()
print(json_file)
get_config_params(json_file)

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


# Check if legal features are selected
command = "SHOW FIELDS FROM Sound_statistics FROM " + database
cursor.execute(command)
fields = str(cursor.fetchall())
fields = fields.replace('(','').replace(')','')
#fields = fields.replace(')','')
available_fields = fields
for a in selected_features:
    if (a in available_fields) == False:
        print("Selected feature is not in database: ", a)
        exit(1)

# Retreive data from the database
feature_vectors = np.zeros((1, len(selected_features)), dtype=float)
vector = np.zeros((1, len(selected_features)), dtype=float)
row = 0
# Create a one dimentional array of feature data and reshape te array later
for record in selected_records:
    # Get the record
    command = "select Recording.sound from Recording where sound_file = %s"
    cursor.execute(command, (record,))
    sound_id = cursor.fetchall()[-1][0]
    print("Sound_id: ", sound_id)
    command = "select Sound.features from Sound where Sound.id = %s"
    cursor.execute(command, (sound_id,))
    feature_id = cursor.fetchall()[-1][0]
    print("Feature id: ", feature_id)
    # Loop through all feature vectors
    feature_vector_cnt = 0
    while feature_id:
        feature = 0
        for a in selected_features:
            command = "select Sound_features."+a+" from Sound_features where Sound_features.id = "+str(feature_id)
            cursor.execute(command)
            vector[0][feature] = float(cursor.fetchall()[-1][0])
            feature +=1
            #feature_vectors[feature_index]=cursor.fetchall()[-1][0]
        feature_vectors = np.append(feature_vectors, vector, axis=0)
        feature_vector_cnt +=1
        command = "select Sound_features.next from Sound_features where Sound_features.id = %s"
        cursor.execute(command, (feature_id,))
        feature_id = cursor.fetchall()[-1][0]
    print(feature_vector_cnt, " vectors added")

# Remove the first empty row
feature_vectors = np.delete(feature_vectors, 0, 0)
# Normalize the values
std = np.std(feature_vectors, axis=0)
mean = np.mean(feature_vectors, axis=0)
norm_vectors = np.divide(np.subtract(feature_vectors,mean),std)

# Save the setup parameters
t = datetime.now()
filename_ext = str(t.year) + "-" + str(t.month) + "-" + str(t.day) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + \
              str(t.second)
filename = "./data/training"+filename_ext+".npy"
# Generate training set for un-supervised training
np.save(filename, feature_vectors)
# Save config parameters
json_filename = get_json_filename()
with open(json_filename) as json_file:
    config_param = json.load(json_file)
    print(config_param)
filename = "./data/config"+filename_ext+".json"
with open(filename, 'w') as json_file:
    json.dump(config_param, json_file)

cursor.close()
cnx.commit()
cnx.close()
