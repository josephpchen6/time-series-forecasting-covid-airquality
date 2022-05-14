import pandas as pd
import pandas as pd
import numpy as np
import os.path

from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from numpy import log
from matplotlib import pyplot

#data is from 5/12/22

countyFips= 27123
cityName= 'Saint Paul'

def case(county):
    global interpolate
    raw = read_csv('uscountycases.csv', header=0, index_col=0,
    parse_dates=True, squeeze=True)
    filter1 = raw[raw['fips'] == county]
    cases1 = filter1['cases']
    perDay = cases1.diff()
    positive = perDay[perDay > 0]
    caseavg = positive.rolling(window = 7)
    rollingavg = caseavg.mean()
    upsample = rollingavg.resample('D').mean()
    interpolate = upsample.interpolate(method='linear')
    interpolate.drop_duplicates(keep='first')

def aq(city):
    global df2
    raw = pd.concat(map(pd.read_csv, ['2022.csv', '2021Q4.csv', '2021Q3.csv', '2021Q2.csv',
    '2021Q1.csv', '2020Q4.csv', '2020Q3.csv', '2020Q2.csv', '2020Q1.csv']))
    filter2 = raw[raw['City'] == city]                  
    o3Filter = filter2[filter2['Specie'] == 'o3']
    o3Sort = o3Filter.drop(['Country', 'City', 'Specie', 'count', 'min', 'max', 'variance'], axis=1)
    o3Sort['Date'] = pd.to_datetime(o3Filter['Date'])
    o3Chron = o3Sort.sort_values(by='Date')
    o3Chron.columns = (['Date', 'o3'])
    o3 = o3Chron.set_index('Date')
                  
    pm10Filter = filter2[filter2['Specie'] == 'pm10']
    pm10Sort = pm10Filter.drop(['Country', 'City', 'Specie', 'count', 'min', 'max', 'variance'], axis=1)
    pm10Sort['Date'] = pd.to_datetime(pm10Filter['Date'])
    pm10Chron = pm10Sort.sort_values(by='Date')
    pm10Chron.columns = (['Date','pm10'])
    pm10 = pm10Chron.set_index('Date')

    no2Filter = filter2[filter2['Specie'] == 'no2']
    no2Sort = no2Filter.drop(['Country', 'City', 'Specie', 'count', 'min', 'max', 'variance'], axis=1)
    no2Sort['Date'] = pd.to_datetime(no2Filter['Date'])
    no2Chron = no2Sort.sort_values(by='Date')
    no2Chron.columns = (['Date', 'no2'])
    no2 = no2Chron.set_index('Date')

    df1 = o3.join(pm10)
    df2 = df1.join(no2)
    df2.drop_duplicates(keep='first')
    
def plot(graph):
    graph.plot()
    pyplot.yscale('log')
    pyplot.show()

case(countyFips)
aq(cityName)
    
both = df2.join(interpolate)
both = both[~both.index.duplicated(keep='first')]
both.to_csv(str(cityName)+'.csv')
plot(both)

