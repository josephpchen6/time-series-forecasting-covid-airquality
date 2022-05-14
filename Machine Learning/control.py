#model without air quality

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

#data is from 3/3/20 to 1/3/22 

rawData = pd.read_csv('Raleigh.csv', header=0, index_col=0,
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
#create model: 1 dense layer

model = keras.Model(
  inputs=[today],
  outputs=[todDense]
  )
#define input and output

model.compile(
  optimizer='adam',loss=keras.losses.MeanSquaredError(),
  metrics=keras.losses.MeanSquaredError()
  )
#compile the model optimizing mean squared error

weights = LambdaCallback(on_epoch_end=lambda batch,
                         logs: [print(todayLayer.get_weights())])
#show layer weights with each epoch

history = model.fit(
  {'CasesToday': dataframe['todcases']},
  {'dense':dataframe['tomcases']},
  epochs=10,
  batch_size=100,
  callbacks = [weights],
  validation_split=.33,
  )
#use today's cases in the dataframe as input, compare to tomorrow's cases in the dataframe as output
#change epoch and batch size if desired

#save model here if desired
#model.save('control')

guess=model.predict({'CasesToday': dataframe['todcases']})
#use model to create a prediction

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
