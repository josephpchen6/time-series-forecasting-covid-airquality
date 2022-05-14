'''
NOTE:
THIS FILE IS SOLELY FOR THE CREATION OF A MODEL DIAGRAM FOR THE CONTROL, NOT FOR ANY MACHINE LEARNING

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

rawData = pd.read_csv('Raleigh.csv', header=0, index_col=0,
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

model = keras.Model(
  inputs=[today],
  outputs=[todDense]
  )

model.compile(
  optimizer='adam',loss=keras.losses.MeanSquaredError(),
  )
#no metrics

history = model.fit(
  {'CasesToday': dataframe['todcases']},
  {'dense':dataframe['tomcases']},
  epochs=1,
  batch_size=1000,
  validation_split=.33,
  )
#only 1 batch/epoch: should take very little time

tf.keras.utils.plot_model(model, 'control_diagram.png')
