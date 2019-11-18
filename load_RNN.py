import tensorflow as tf
import numpy as np
import wind_data
import os
read_from_file = open('api_keys.txt', 'r')
google_api_key = read_from_file.read().split("\n")[0]
os.environ["GOOGLE_API_KEY"] = google_api_key
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


#coordFinal = wind_data.get_coord(str(Lat), str(Long), time=timeDays)


def load_and_compute(lat, long, alt, month, year):
    input = [[lat, long, alt, month, year]]
    input = np.array(input)
    input = np.reshape(input, (input.shape[0], 1, input.shape[1]))

    #load
    filename = '/var/www/acid/code/Acid/RainAcidRainModel.h5'
    print(filename)
    model = tf.keras.models.load_model(filename)
    result = model.predict(input)
    print(result)
    return result

#load_and_compute(32.1145231,-110.6911934,929.6,12.80645161,1951)

write_to_file = open("predict.txt", "w+")
write_to_file.write('var result = ')
#write_to_file.write(str(load_and_compute(coordFinal, Alt, month, year)[0][0][0]))
write_to_file.write(str(load_and_compute(32.1145231,-110.6911934,929.6,12.80645161,1951)[0][0][0]))
write_to_file.close()
