import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import pprint as pp
from sklearn.model_selection import train_test_split

# load dataset
path = os.getcwd()
if sys.platform == 'win32':
    path += "\\weatherdata"
elif sys.platform == 'linux':
    path += "/weatherdata"
data3 = pd.DataFrame()
for filename in os.listdir(path):
    # changed the date to month and year
    column_names = ['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'MONTH', 'YEAR', 'PRECIPITATION']
    raw_data = pd.read_csv(path+'/'+filename, names=column_names, na_values="?", comment= "\t", sep=",",
                           skipinitialspace=True)
    data = raw_data.copy()
    data = data.drop(columns=['STATION', 'NAME'])
    # after processing
    data3 = pd.concat([data3, data])
pp.pprint(data3)

# This requires scikit-learn. Get it Bill.
train_dataset, test_dataset = train_test_split(data3, train_size=0.8)

'''
sns.pairplot(train_dataset[['LATITUDE', 'LONGITUDE', 'PRECIPITATION']], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("PRECIPITATION")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('PRECIPITATION')
test_labels = test_dataset.pop('PRECIPITATION')
'''

def norm(x):
    # return (x - train_stats['mean']) / train_stats['std']
    return x


train_X = train_dataset[['LATITUDE', 'LONGITUDE', 'ELEVATION', 'MONTH', 'YEAR']]
train_Y = train_dataset['PRECIPITATION']
test_X = test_dataset[['LATITUDE', 'LONGITUDE', 'ELEVATION', 'MONTH', 'YEAR']]
test_Y = test_dataset['PRECIPITATION']

# load model
def build_model():
    model = keras.Sequential()
    '''
    model.add(keras.layers.Dense(64, input_shape=[5]))
    model.add(keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(1, activation=tf.nn.relu))
    '''
    model.add(tf.keras.layers.Embedding(64, input_shape=[5]))
    model.add(tf.keras.layers.LSTM(1024))
    model.add(tf.keras.layers.LSTM(1024))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

    optimizer = tf.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

example_batch = train_X[:10]
print(example_batch)
example_result = model.predict(example_batch)
print(example_result)

#train
history = model.fit(train_X,
                    train_Y,
                    batch_size=64,
                    epochs=50)

model.save('our_model_cue_USSR_theme.h5')

#test
