import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import pprint as pp


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
    data = data.dropna()
    data = data.drop(0)
    data = data.drop('STATION', axis=1)
    data = data.drop('NAME', axis=1)
    # after processing
    data3 = pd.concat([data3, data])
pp.pprint(data3)
print(data3.shape)

train_dataset = data3.sample(frac=0.8,random_state=0)
test_dataset = data3.drop(train_dataset.index)

sns.pairplot(train_dataset[['LATITUDE', 'LONGITUDE', 'PRECIPITATION']], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("PRECIPITATION")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('PRECIPITATION')
test_labels = test_dataset.pop('PRECIPITATION')


def norm(x):
    # return (x - train_stats['mean']) / train_stats['std']
    return x


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# load model
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_shape=[4]))
    model.add(keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(1024, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(1, activation=tf.nn.relu))

    optimizer = tf.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

#train


#test