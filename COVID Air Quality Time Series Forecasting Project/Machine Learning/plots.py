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
todayLayer = layers.Dense(1, activation = 'relu',
                        kernel_constraint='NonNeg'
                          )
todDense=todayLayer(today)

o3 = keras.Input(shape = (1), name='OzoneMeasurement')
o3Layer = layers.Dense(1, activation = 'relu',
                       kernel_constraint='NonNeg'
                       )
o3Dense=o3Layer(o3)

pm10 = keras.Input(shape = (1), name='PM10Measurement')
pm10Layer = layers.Dense(1, activation = 'relu',
                         kernel_constraint='NonNeg'
                         )
pm10Dense=pm10Layer(pm10)

no2 = keras.Input(shape = (1), name='NO2Measurement')
no2Layer = layers.Dense(1, activation = 'relu',
                        kernel_constraint='NonNeg'
                        )
no2Dense=no2Layer(no2)

merge = layers.concatenate([todDense, o3Dense, pm10Dense, no2Dense])
predictions = layers.Dense(1, name='Prediction',activation = 'relu', kernel_constraint='NonNeg',
                           kernel_initializer=keras.initializers.TruncatedNormal(mean=1, stddev=0.05))
predictionsDense=predictions(merge)

model = keras.Model(
  inputs=[today, o3, pm10, no2],
  outputs=[predictionsDense]
  )

model.compile(
  optimizer='RMSprop',loss=keras.losses.MeanSquaredError(),
  metrics=keras.losses.MeanSquaredError()
  )

weights = LambdaCallback(on_epoch_end=lambda batch,
                         logs: [print(predictions.get_weights()),
                         print(todayLayer.get_weights()),
                        print(o3Layer.get_weights()),
                        print(pm10Layer.get_weights()),
                        print(no2Layer.get_weights())])

history = model.fit(
  {'CasesToday': dataframe['todcases'], 'OzoneMeasurement': dataframe['o3'],
   'PM10Measurement': dataframe['pm10'], 'NO2Measurement': dataframe['no2']},
  {'Prediction':dataframe['tomcases']},
  epochs=2,
  batch_size=100,
  
  callbacks = [weights],
  validation_split=.33
  )

guess=model.predict({'CasesToday': dataframe['todcases'], 'OzoneMeasurement': dataframe['o3'],
   'PM10Measurement': dataframe['pm10'], 'NO2Measurement': dataframe['no2']})

plt.figure(dpi=1000)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Sample Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set','Validation Dataset'],loc='upper left')
plt.savefig('loss.png')

plt.figure(dpi=1000)
plt.plot(guess)
dataframe.reset_index(drop=True, inplace=True)
plt.plot(dataframe['tomcases'])
plt.title('Sample Model Prediction')
plt.xlabel('Days Since March 3, 2020')
plt.ylabel('Cases')
plt.legend(['Prediction','Actual'],loc='upper left')
plt.savefig('predictions.png')

