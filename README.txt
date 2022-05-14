My attempt at using Keras and machine learning to create a model which tests the correlation between air quality and COVID cases in the US.
The model uses 3 air quality measurements (ozone, pm10, no2) as well as cases from the previous day to make a prediction

Data used was from 3/3/20 to 5/12/22.
COVID case data was downloaded from the New York Times COVID dataset (https://github.com/nytimes/covid-19-data)
Air quality case data was downloaded from the World Air Quality Index (https://aqicn.org/data-platform/covid19/)
If you would like to use more recent data, download both datasets and use DataImporter.py to update the data.

model.py and control.py are the actual ML files

Suggestions:
- Use cases from multiple days in the past (I simply did 1 day)
- Use standard deviation from the mean instead of raw case numbers (this would make the model more widely applicable across cities)
- Include cities other than those which contained all 3 air quailty measurements
- Loss function that is superior to MSE, one that prioritizes nearly correct guesses (i.e. ones that are off by <5%)
