"""Persistent (naive) model. You could use this as a negative control, though I didn't."""

import pandas as pd
import numpy as np
from matplotlib import pyplot

#data is from 5/12/22

def naive (file):
    """Runs persistent model (prediction = value at t - 1)."""
    raw = pd.read_csv(f'{file}.csv', header = 0, index_col=0,
    parse_dates=True, squeeze=True)
    t = raw["cases"]
    tplus1 = raw.shift(1)["cases"]
    diff = t - tplus1
    abs_diff = abs(diff)
    square = diff*diff
    rmse = round(square.mean(),2)
    print(f"MSE: {rmse}")
    diff_mean = round((abs_diff.mean()),2)
    print(f"MAE: {diff_mean}")
    pyplot.plot(absdiff)
    pyplot.title("Mean Absolute Error v. Date")
    pyplot.show()
    pyplot.title("Mean Squared Error v. Date")
    pyplot.plot(square)
    pyplot.show()

naive("Albuquerque")
