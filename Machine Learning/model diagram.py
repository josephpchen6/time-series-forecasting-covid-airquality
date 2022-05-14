'''
NOTE:
THIS FILE IS SOLELY FOR THE CREATION OF A MODEL DIAGRAM, NOT FOR ANY MACHINE LEARNING

For some reason, I could not save a .png of a model diagram when said model is compiled with MSE metrics
So this code removes the MSE metrics (along with other parts to speed up the process)

Mac Users:
I also completed this project on Mac, where pydot (what is needed to plot the model) is much more difficult to install
I just did this part on a Windows computer, which I would recommend over installing pydot
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

rawData = pd.read_csv('Boise.csv', header=0, index_col=0,
parse_dates=True, squeeze=True)
dataframe = pd.concat([rawData, rawData['cases'].shift(-1)], axis=1)
dataframe=dataframe.dropna()
dataframe.columns=['o3', 'pm10', 'no2', 'todcases', 'tomcases']
dataframe.to_numpy()

today = keras.Input(shape = (1), name='CasesToday')
todayLayer = layers.Dense(1,activation = 'relu', kernel_constraint='NonNeg',
                          kernel_initializer=keras.initializers.TruncatedNormal(mean=1, stddev=0.05),
                          )
todDense=todayLayer(today)

o3 = keras.Input(shape = (1), name='OzoneMeasurement')
o3Layer = layers.Dense(1, activation = 'sigmoid', kernel_constraint='NonNeg',
                       kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
                       )
o3Dense=o3Layer(o3)

pm10 = keras.Input(shape = (1), name='PM10Measurement')
pm10Layer = layers.Dense(1, activation = 'sigmoid', kernel_constraint='NonNeg',
                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
                         )
pm10Dense=pm10Layer(pm10)

no2 = keras.Input(shape = (1), name='NO2Measurement')
no2Layer = layers.Dense(1, activation = 'sigmoid', kernel_constraint='NonNeg',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
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
  optimizer='adam',loss=keras.losses.MeanSquaredError(),
  )

history = model.fit(
  {'CasesToday': dataframe['todcases'], 'OzoneMeasurement': dataframe['o3'],
   'PM10Measurement': dataframe['pm10'], 'NO2Measurement': dataframe['no2']},
  {'Prediction':dataframe['tomcases']},
  epochs=1,
  batch_size=1000,
  validation_split=.33,
  )
#only 1 batch/epoch: should take very little time

tf.keras.utils.plot_model(model, 'diagram.png')
