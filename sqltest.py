import mysql.connector
from datetime import date, datetime, timedelta
import time

from mysql.connector import errorcode

DB_NAME = "sound"

print(mysql.connector.__version__)
cnx = mysql.connector.connect(user='parallels', password='ubuntuerbra', host='localhost', database='sound')
cursor = cnx.cursor()
try:
    cursor.execute("USE {}".format(DB_NAME))
except mysql.connector.Error as err:
    print("Database {} does not exists.".format(DB_NAME))
    print(err)
    exit(1)

# Delete all existing records
try:
    cursor.execute("delete from Record where sound_file = './TestSoundFile.wav'")
except mysql.connector.Error as err:
    print(err)
    exit(1)

# Create a few new sound records
for i in range(4):
    date = str(datetime.now())
    date = date.split(".")[0]
    add_sound_record = ("insert into Record"
                        "(date, sound_file) "
                        "values (%s, %s)")
    sound_record_data = ( date, "./TestSoundFile.wav")
    try:
        cursor.execute(add_sound_record, sound_record_data)
    except mysql.connector.Error as err:
        print(err)
        exit(1)
    time.sleep(1)
cnx.commit()

# Replace one element in a record
try:
    cursor.execute("update Record set Record.sound_file='./Tullogtoys.txt' where Record.id = 16")
except mysql.connector.Error as err:
    print(err)
    exit(1)

# Get element(s) in a record
try:
    cursor.execute("select id from Record")
    print(cursor)
    for (id) in cursor:
        print("Id:", id)
except mysql.connector.Error as err:
    print(err)
    exit(1)

cursor.close()
cnx.close()

