"""Generate a synthetic PV dataset for testing the pipeline."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
n = 2000
timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)]

hour = np.array([t.hour for t in timestamps])
day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])

# Solar generation: only during daylight hours
solar_base = np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) * np.sin(np.pi * day_of_year / 365)
solar_generation = solar_base * 100 + np.random.normal(0, 5, n)
solar_generation = np.clip(solar_generation, 0, None)

air_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + 5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1, n)
dew_point = air_temp - np.random.uniform(2, 8, n)
rel_humidity = np.clip(60 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5, n), 0, 100)
sun_azimuth = np.where(hour < 6, 0, np.where(hour > 20, 0, 180 + 180 * np.sin(np.pi * (hour - 6) / 14))) + np.random.normal(0, 5, n)
cloud_opacity = np.clip(np.random.beta(0.5, 2, n) * 100, 0, 100)
dni = np.clip(solar_base * 800 * (1 - cloud_opacity / 200) + np.random.normal(0, 20, n), 0, 1000)
ghi = np.clip(solar_base * 600 + np.random.normal(0, 15, n), 0, 800)
precipitable_water = np.clip(20 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, n), 0, None)
wind_speed = np.abs(np.random.weibull(2, n) * 5)

df = pd.DataFrame({
    "Timestamp": timestamps,
    "SolarGeneration": solar_generation,
    "AirTemperature": air_temp,
    "DewPointTemperature": dew_point,
    "RelativeHumidity": rel_humidity,
    "SunAzimuth": sun_azimuth,
    "cloud_opacity": cloud_opacity,
    "dni": dni,
    "ghi": ghi,
    "precipitable_water": precipitable_water,
    "wind_speed_10m": wind_speed,
})

df.to_csv("data/pv_dataset.csv", index=False)
print(f"Saved {n} rows to data/pv_dataset.csv")
