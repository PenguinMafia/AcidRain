import tensorflow as tf
import numpy as np



def load_and_compute(lat, long, alt, month, year):
    input = [[lat, long, alt, month, year]]
    input = np.array(input)
    input = np.reshape(input, (input.shape[0], 1, input.shape[1]))

    #load
    filename = 'C:\\Users\\Kitsunebula\\Desktop\\AcidRain\\AcidRainModel.h5'
    print(filename)
    model = tf.keras.models.load_model(filename)
    result = model.predict(input)
    print(result)

load_and_compute(32.1145231,-110.6911934,929.6,12.80645161,1951)