import tensorflow as tf
import numpy as np
import wind_data
import os
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBtF0YdfyMRZyE1Cy5k0WgRDYcTqdBQar4'
import geocoder


g = geocoder.ip('me')
geo = g.latlng
Lat = geo[0]
Long = geo[1]
g = geocoder.google([Lat, Long], method='elevation')
Alt = g.feet
month = 11.5
year = 2019
timeDays = 30


coordFinal = wind_data.get_coord(str(Lat), str(Long), time=timeDays)


def load_and_compute(lat, long, alt, month, year):
    input = [[float(lat), float(long), float(alt), month, year]]
    input = np.array(input, dtype=tf.float32)
    input = np.reshape(input, (input.shape[0], 1, input.shape[1]))

    #load
    filename = 'C:\\Users\\Kitsunebula\\Desktop\\AcidRain\\AcidRainModel.h5'
    print(filename)
    model = tf.keras.models.load_model(filename)
    result = model.predict(input)
    print(result)
    return result

#load_and_compute(32.1145231,-110.6911934,929.6,12.80645161,1951)

write_to_file = open("predict.txt", "w+")
write_to_file.write('var result = ')
write_to_file.write(str(load_and_compute(coordFinal, Alt, month, year)[0][0][0]))
write_to_file.close()