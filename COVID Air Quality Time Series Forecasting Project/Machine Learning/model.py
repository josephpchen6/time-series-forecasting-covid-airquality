#model with air quality

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

#data is from 3/3/20 to 5/12/22 

rawData = pd.read_csv('Boise.csv', header=0, index_col=0,
parse_dates=True, squeeze=True)
dataframe = pd.concat([rawData, rawData['cases'].shift(-1)], axis=1)
dataframe=dataframe.dropna()
dataframe.columns=['o3', 'pm10', 'no2', 'todcases', 'tomcases']
dataframe.to_numpy()
#import data from .csv


today = keras.Input(shape = (1), name='CasesToday')
todayLayer = layers.Dense(1,activation = 'relu', kernel_constraint='NonNeg',
                          kernel_initializer=keras.initializers.TruncatedNormal(mean=1, stddev=0.05),
                          )
todDense=todayLayer(today)
#input layer: today's cases
#note the relu activation and kernel value

o3 = keras.Input(shape = (1), name='OzoneMeasurement')
o3Layer = layers.Dense(1, activation = 'sigmoid', kernel_constraint='NonNeg',
                       kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
                       )
o3Dense=o3Layer(o3)
#input layer: ozone
#note the sigmoid activation and kernel value

pm10 = keras.Input(shape = (1), name='PM10Measurement')
pm10Layer = layers.Dense(1, activation = 'sigmoid', kernel_constraint='NonNeg',
                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
                         )
pm10Dense=pm10Layer(pm10)
#input layer: pm10
#note the sigmoid activation and kernel value

no2 = keras.Input(shape = (1), name='NO2Measurement')
no2Layer = layers.Dense(1, activation = 'sigmoid', kernel_constraint='NonNeg',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
                        )
no2Dense=no2Layer(no2)
#input layer: no2
#note the sigmoid activation and kernel value

merge = layers.concatenate([todDense, o3Dense, pm10Dense, no2Dense])
predictions = layers.Dense(1, name='Prediction',activation = 'relu', kernel_constraint='NonNeg',
                           kernel_initializer=keras.initializers.TruncatedNormal(mean=1, stddev=0.05))
predictionsDense=predictions(merge)
#layers are concatenated and merged to create a single output

model = keras.Model(
  inputs=[today, o3, pm10, no2],
  outputs=[predictionsDense]
  )
#define inputs and output

model.compile(
  optimizer='adam',loss=keras.losses.MeanSquaredError(),
  metrics=keras.losses.MeanSquaredError()
  )
#compile the model optimizing mean squared error

weights = LambdaCallback(on_epoch_end=lambda batch,
                         logs: [print(predictions.get_weights()),
                         print(todayLayer.get_weights()),
                        print(o3Layer.get_weights()),
                        print(pm10Layer.get_weights()),
                        print(no2Layer.get_weights())])
#show layer weights with each epoch

history = model.fit(
  {'CasesToday': dataframe['todcases'], 'OzoneMeasurement': dataframe['o3'],
   'PM10Measurement': dataframe['pm10'], 'NO2Measurement': dataframe['no2']},
  {'Prediction':dataframe['tomcases']},
  epochs=4,
  batch_size=100,
  callbacks = [weights],
  validation_split=.33,
  )
#use cases and air quality in the dataframe as input, compare to tomorrow's cases in the dataframe as output
#change epoch and batch size if desired

guess=model.predict({'CasesToday': dataframe['todcases'], 'OzoneMeasurement': dataframe['o3'],
   'PM10Measurement': dataframe['pm10'], 'NO2Measurement': dataframe['no2']})
#use model to create a prediction

#save model here if desired
#model.save('model')

plt.figure(dpi=1000)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Sample Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set','Validation Dataset'],loc='upper left')
plt.savefig('loss.png')
#plot: loss every epoch

plt.figure(dpi=1000)
plt.plot(guess)
dataframe.reset_index(drop=True, inplace=True)
plt.plot(dataframe['tomcases'])
plt.legend(['predicted', 'acutal'], loc = 'upper right')
plt.savefig('predictions.png')
#plot: model predictions v. actual
