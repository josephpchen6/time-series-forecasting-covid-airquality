"""Import, filter, combine, and save COVID and air quality data to county-based CSV file."""

import pandas as pd

class main():
    """Sets shared variables across functions."""
    
    case_df = pd.DataFrame()
    aq_df = pd.DataFrame()
    waqi_string = "raw_data/waqi-covid19-airqualitydata-"
    
    def case(county_fips):
        """Import COVID case data."""
        raw_aq = pd.concat(map(pd.read_csv, ["raw_data/us-counties-2020.csv",
        "raw_data/us-counties-2021.csv","raw_data/us-counties-2022.csv"]))
        county_filter = raw_case[raw_case["fips"] == county_fips]
        daily_cases = county_filter["cases"].diff()
        resampled = daily_cases[daily_cases > 0].interpolate(method="time")
        main.case_df = resampled.rolling(window = 7).mean().to_frame()

    def aq(city):
        """Import air quality data."""
        raw_aq = pd.concat(map(pd.read_csv, [f"{waqi_string}2022.csv", f"{waqi_string}2021Q4.csv",
        f"{waqi_string}2021Q3.csv", f"{waqi_string}2021Q2.csv",f"{waqi_string}2021Q1.csv", f"{waqi_string}2020Q4.csv",
        f"{waqi_string}2020Q3.csv", f"{waqi_string}2020Q2.csv", f"{waqi_string}2020Q1.csv"]))
        aq_filter = raw_aq[raw_aq["City"] == city]
        #make o3 df
        o3_filter = aq_filter[aq_filter["Specie"] == "o3"].drop(
        ["Country", "City", "Specie", "count", "min", "max", "variance"], axis=1
        )
        o3_filter["Date"] = pd.to_datetime(o3_filter["Date"])
        o3_chron = o3_filter.sort_values(by="Date")
        o3_chron.columns = (["Date", "o3"])
        o3_df = o3_chron.set_index("Date")
        #make pm10 df     
        pm10_filter = aq_filter[aq_filter["Specie"] == "pm10"].drop(
        ["Country", "City", "Specie", "count", "min", "max", "variance"], axis=1
        )
        pm10_filter["Date"] = pd.to_datetime(pm10_filter["Date"])
        pm10_chron = pm10_filter.sort_values(by="Date")
        pm10_chron.columns = (["Date","pm10"])
        pm10_df = pm10_chron.set_index("Date")
        #make no2 df
        no2_filter = aq_filter[aq_filter["Specie"] == "no2"].drop(
        ["Country", "City", "Specie", "count", "min", "max", "variance"], axis=1
        )
        no2_filter["Date"] = pd.to_datetime(no2_filter["Date"])
        no2_chron = no2_filter.sort_values(by="Date")
        no2_chron.columns = (["Date", "no2"])
        no2_df = no2_chron.set_index("Date")
        #combine dfs
        main.aq_df = o3_df.join(pm10_df.join(no2_df))

    def main(county_fips, county_name):
        """Runs the two functions."""
        main.case(county_fips)
        main.aq(county_name)
        final_df = main.case_df.join(main.aq_df)
        final_df = final_df[~final_df.index.duplicated(keep="first")]
        final_df.to_csv(f"county_data/{county_name}.csv")

main.main()

