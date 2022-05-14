#Persistent (naive) model. You could use this as a negative control, though I didn't.

import pandas as pd
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import log
from matplotlib import pyplot

#data is from 5/12/22
def naive (file):
    dataset = f'{file}.csv'
    raw = pd.read_csv(dataset, header = 0, index_col=0,
                      parse_dates=True, squeeze=True)
    t = raw['cases']
    tplus1 = raw.shift(1)['cases']
    diff = t - tplus1
    absdiff = abs(diff)
    square = diff*diff
    rmse = round(square.mean(),2)
    print(f'MSE: {rmse}')
    diffmean = round((absdiff.mean()),2)
    print(f'MAE: {diffmean}')
    pyplot.plot(absdiff)
    pyplot.title('Mean Absolute Error v. Date')
    pyplot.show()
    pyplot.title('Mean Squared Error v. Date')
    pyplot.plot(square)
    pyplot.show()

naive('Albuquerque')
