import geopandas as gpd
import pandas as pd
import openmeteo_requests
import time
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np
import openmeteo_requests

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

# Stull approx. for wet-bulb temperature
def wetbulb(T, RH):
    return (
        T*np.arctan(0.151977*np.sqrt(RH+8.313659))
        + np.arctan(T+RH)
        - np.arctan(RH-1.676331)
        + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH)
        - 4.686035
    )

def fetch_climate(lat, lon):

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"],
        "hourly": "relative_humidity_2m",
        "timezone": "UTC"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()

    daily_df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temp": daily.Variables(0).ValuesAsNumpy(),
        "wind": daily.Variables(1).ValuesAsNumpy(),
        "precip": daily.Variables(2).ValuesAsNumpy()
    })

    hourly = response.Hourly()

    hourly_df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "humidity": hourly.Variables(0).ValuesAsNumpy()
    })

    hourly_df["date"] = hourly_df["datetime"].dt.date

    humidity_daily = (
        hourly_df
        .groupby("date")["humidity"]
        .mean()
        .reset_index()
    )

    daily_df["date"] = daily_df["date"].dt.date

    daily_df = daily_df.merge(humidity_daily, on="date", how="left")

    return daily_df

gdf = gpd.read_file('/Users/jaydenudall/Desktop/datathon data/CONUS_CLIMATE_DIVISIONS/GIS.OFFICIAL_CLIM_DIVISIONS.shp')
climdiv = gdf.copy().to_crs("EPSG:4326")
points = climdiv.copy()
points["geometry"] = climdiv.geometry.representative_point()
points["lat"] = points.geometry.y
points["lon"] = points.geometry.x

print(points.shape)

records = []
i = 0
for _, row in points.iterrows():

    df = fetch_climate(row["lat"], row["lon"])

    df["wetbulb"] = wetbulb(df["temp"], df["humidity"])

    records.append({
        "CLIMDIV": row["CLIMDIV"],
        "wetbulb": df["wetbulb"].mean(),
        "wind": df["wind"].mean(),
        "precip": df["precip"].mean()
    })

    print(f"Loop {i}, Processed {row['CLIMDIV']}")
    i += 1

    time.sleep(1.1)  # avoid rate limiting

climate_df = pd.DataFrame(records)