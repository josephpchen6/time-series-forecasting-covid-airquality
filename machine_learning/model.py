"""Model with air quality."""

import pandas as pd
import keras

from matplotlib import pyplot as plt
from keras import layers
from keras.callbacks import LambdaCallback

class main():
  """Sets shared variables across ML and plotting functions."""
  model_df = ""
  model = ""
  history = ""
  file = ""

  def model_creation(file, epochs, batch_size):
    """Creates control according to file, with paramaters for the epochs and batch size."""
    main.file = file
    raw_county = pd.read_csv(f"county_data/{main.file}.csv", header=0, index_col=0,
    parse_dates=True, squeeze=True)
    main.model_df = pd.concat([raw_county, raw_county["cases"].shift(-1)], axis=1).dropna()
    main.model_df.columns = ["o3", "pm10", "no2", "todcases", "tomcases"]
    #import data from .csv
    today_input = keras.Input(shape = (1), name="cases_today")
    today_dense = layers.Dense(1,activation="relu", kernel_constraint="NonNeg",
    kernel_initializer=keras.initializers.TruncatedNormal(mean=1, stddev=0.05),
    )
    today_layer = today_dense(today_input)
    #input layer: today's cases with relu activation and kernel constraint
    o3_input = keras.Input(shape = (1), name="ozone_measurement")
    o3_dense = layers.Dense(1, activation="sigmoid", kernel_constraint="NonNeg",
    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
    )
    o3_layer = o3_dense(o3_input)
    #input layer: ozone with sigmoid activation and kernel constraint
    pm10_input = keras.Input(shape = (1), name="pm10_measurement")
    pm10_dense = layers.Dense(1, activation="sigmoid", kernel_constraint="NonNeg",
    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
    )
    pm10_layer = pm10_dense(pm10_input)
    #input layer: pm10 with sigmoid activation and kernel constraint
    no2_input = keras.Input(shape = (1), name="no2_measurement")
    no2_dense = layers.Dense(1, activation = "sigmoid", kernel_constraint="NonNeg",
    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05),
    )
    no2_layer = no2_dense(no2_input)
    #input layer: no2 with sigmoid activation and kernel constraint
    concatenated_layers = layers.concatenate([today_layer, o3_layer, pm10_layer, no2_layer])
    dense_predictions = layers.Dense(1, name="prediction", activation = "relu", kernel_constraint="NonNeg",
    kernel_initializer=keras.initializers.TruncatedNormal(mean=1, stddev=0.05))
    prediction_layer = dense_predictions(concatenated_layers)
    #layers are concatenated and merged to create a single output
    main.model = keras.Model(
      inputs=[today_input, o3_input, pm10_input, no2_input],
      outputs=[prediction_layer]
      )
    #define inputs and output
    main.model.compile(
      optimizer="adam",loss=keras.losses.MeanSquaredError(),
      metrics=keras.losses.MeanSquaredError()
      )
    #compile the model optimizing mean squared error
    weights = LambdaCallback(on_epoch_end=lambda batch,
      logs: [print(dense_predictions.get_weights()), print(today_dense.get_weights()),
      print(o3_dense.get_weights()), print(pm10_dense.get_weights()), print(no2_dense.get_weights())])
    #show layer weights with each epoch
    main.history = main.model.fit(
      {"cases_today": main.model_df["todcases"], "ozone_measurement": main.model_df["o3"],
      "pm10_measurement": main.model_df["pm10"], "no2_measurement": main.model_df["no2"]},
      {"prediction":main.model_df["tomcases"]},
      epochs=epochs, batch_size=batch_size, callbacks = [weights], validation_split=.33,
      )
    #use cases and air quality in the dataframe as input, compare to tomorrow's cases in the dataframe as output\
    main.model.save(f"models/{main.file}_model")

  def model_loss_plot():
    """Create a graph of the control's loss over time."""
    plt.figure(dpi=1000)
    plt.plot(main.history.history["loss"])
    plt.plot(main.history.history["val_loss"])
    plt.title("Sample Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training Set","Validation Dataset"],loc="upper left")
    plt.savefig(f"diagrams/{main.file}_model_loss.png")

  def model_prediction_graph():
    """Create a graph of the predictions of the model vs. the actual number of cases."""
    guess = main.model.predict({"cases_today": main.model_df["todcases"], "ozone_measurement": main.model_df["o3"],
    "pm10_measurement": main.model_df["pm10"], "no2_measurement": main.model_df["no2"]})
    plt.figure(dpi=1000)
    plt.plot(guess)
    main.model_df.reset_index(drop=True, inplace=True)
    plt.plot(main.model_df["tomcases"])
    plt.legend(["predicted", "acutal"], loc = "upper right")
    plt.savefig(f"diagrams/{main.file}_model_predictions.png")

main.model_creation()
main.model_loss_plot()
main.model_prediction_graph()