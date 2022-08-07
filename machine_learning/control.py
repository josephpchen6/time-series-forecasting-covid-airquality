#model without air quality

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

class main():

  control_df = ""
  model = ""
  history = ""
  file = ""

  def model_creation(file, epochs, batch_size):

    main.file = file
    raw_county = pd.read_csv(f"county_data/{main.file}.csv", header=0, index_col=0,
    parse_dates=True, squeeze=True)
    main.control_df = pd.concat([raw_county, raw_county["cases"].shift(-1)], axis=1).dropna()
    main.control_df.columns = ["o3", "pm10", "no2", "todcases", "tomcases"]
    #import data from .csv

    today_input = keras.Input(shape = (1), name="cases_today")
    today_layer = layers.Dense(1, activation = "relu", kernel_constraint="NonNeg",
    kernel_initializer = keras.initializers.TruncatedNormal(mean=1, stddev=0.05),
    )
    today_dense = today_layer(today_input)
    #create model: 1 dense layer

    main.model = keras.Model(
      inputs=[today_input],
      outputs=[today_dense]
      )
    #define input and output

    main.model.compile(
      optimizer="adam",loss=keras.losses.MeanSquaredError(),
      metrics=keras.losses.MeanSquaredError()
      )
    #compile the model optimizing mean squared error

    main.weights = LambdaCallback(on_epoch_end=lambda batch,
                            logs: [print(today_layer.get_weights())])
    #show layer weights with each epoch

    main.history = main.model.fit(
      {"cases_today": main.control_df["todcases"]},
      {"dense": main.control_df["tomcases"]},
      epochs=epochs,
      batch_size=batch_size,
      callbacks=[main.weights],
      validation_split=.33,
      )
    #use today's cases in the dataframe as input, compare to tomorrow's cases in the dataframe as output
    #change epoch and batch size if desired

    main.model.save(f"models/{main.file}_control")

  def loss_plot():
    #use model to create a prediction

    plt.figure(dpi=1000)
    plt.plot(main.history.history["loss"])
    plt.plot(main.history.history["val_loss"])
    plt.title("Sample Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training Set","Validation Dataset"],loc="upper left")
    plt.savefig(f"diagrams/{main.file}_control_loss.png")
    #plot: loss every epoch

  def prediction_plot():
    guess = main.model.predict({"cases_today": main.control_df["todcases"]})
    plt.figure(dpi=1000)
    plt.plot(guess)
    main.control_df.reset_index(drop=True, inplace=True)
    plt.plot(main.control_df["tomcases"])
    plt.legend(["predicted", "acutal"], loc = "upper right")
    plt.savefig(f"diagrams/{main.file}_control_predictions.png")
    #plot: model predictions v. actual

main.model_creation()
main.loss_plot()
main.prediction_plot()