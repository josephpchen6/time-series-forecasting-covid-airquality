My attempt (as a high school senior) to use Keras to create a model which tests the correlation between air quality and COVID cases in the US. The model uses 3 air quality measurements (ozone, pm10, no2) as well as cases from the previous day to make a prediction.

COVID case data was downloaded from the New York Times COVID dataset
(https://github.com/nytimes/covid-19-data).
Air quality case data was downloaded from the World Air Quality Index
(https://aqicn.org/data-platform/covid19/).

Download/clone the NY Times COVID dataset and move the us-counties-202x files to raw_data.
Download all files from AQI and move into raw_data.
Use data_importer.py to get the required data.

Suggestions for improvement:
- Use cases from multiple days in the past (I simply did 1 day, which makes the model not too much better than a persistence).
- Use standard deviation from the mean instead of raw case numbers (this would make the model more widely applicable across cities with higher/lower populations).
- Include cities other than those which contained all 3 air quailty measurements (more data).
- Custom loss function which would be superior to MSE; one that prioritizes nearly correct guesses (i.e. ones that are off by <5%;
currently the model is essentially using the previous day's cases as a prediction. As per-day changes in cases are relatively small, this optimizes MSE, but this isn't ideal.).

Updated August 2022 to display better coding practices.