"""Negative control (model without air quality)."""

import pandas as pd
import keras

from matplotlib import pyplot as plt
from keras import layers
from keras.callbacks import LambdaCallback
from keras.utils.vis_utils import plot_model

class main():
  """Sets shared variables across ML and plotting functions."""
  control_df = ""
  control = ""
  history = ""
  file = ""

  def control_creation(file, epochs, batch_size):
    """Creates control according to file, with paramaters for the epochs and batch size."""
    main.file = file
    raw_county = pd.read_csv(f"county_data/{main.file}.csv", header=0, index_col=0,
    parse_dates=True, squeeze=True)
    main.control_df = pd.concat([raw_county, raw_county["cases"].shift(-1)], axis=1).dropna()
    main.control_df = main.control_df.drop(["o3", "pm10", "no2"], axis=1)
    main.control_df.columns = ["todcases", "tomcases"]
    #import data from .csv
    today_input = keras.Input(shape = (1), name="cases_today")
    today_dense = layers.Dense(1, activation = "relu", kernel_constraint="NonNeg",
    kernel_initializer = keras.initializers.TruncatedNormal(mean=1, stddev=0.05),
    )
    today_layer = today_dense(today_input)
    #create control: 1 dense layer
    main.control = keras.Model(
      inputs=[today_input],
      outputs=[today_layer]
      )
    #define input and output
    main.control.compile(
      optimizer="adam",loss=keras.losses.MeanSquaredError(),
      metrics=keras.losses.MeanSquaredError()
      )
    #compile the control optimizing mean squared error
    main.weights = LambdaCallback(on_epoch_end=lambda batch,
                            logs: [print(today_dense.get_weights())])
    #show layer weights with each epoch
    main.history = main.control.fit(
      {"cases_today": main.control_df["todcases"]},
      {"dense": main.control_df["tomcases"]},
      epochs=epochs,
      batch_size=batch_size,
      callbacks=[main.weights],
      validation_split=.33,
      )
    #use today's cases in the dataframe as input, compare to tomorrow's cases in the dataframe as output
    main.control.save(f"models/{main.file}_control")

  def control_loss_plot():
    """Create a graph of the control's loss over time."""
    plt.figure(dpi=1000)
    plt.plot(main.history.history["loss"])
    plt.plot(main.history.history["val_loss"])
    plt.title("Sample Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training Set","Validation Dataset"],loc="upper left")
    plt.savefig(f"diagrams/{main.file}_control_loss.png")

  def control_prediction_plot():
    """Create a graph of the predictions of the control vs. the actual number of cases."""
    guess = main.control.predict({"cases_today": main.control_df["todcases"]})
    plt.figure(dpi=1000)
    plt.plot(guess)
    main.control_df.reset_index(drop=True, inplace=True)
    plt.plot(main.control_df["tomcases"])
    plt.legend(["predicted", "acutal"], loc = "upper right")
    plt.savefig(f"diagrams/{main.file}_control_predictions.png")

  def control_image():
    """
    Plot a diagram of the control's architecture. I could not install graphviz on Mac,
    so I did this part on a Windows computer (which you can do by importing the trained model,
    and solely running this function).
    """
    plot_model(main.control, f"models/{main.file}_model_diagram.png")

main.control_creation()
main.control_loss_plot()
main.control_prediction_plot()
main.control_image()