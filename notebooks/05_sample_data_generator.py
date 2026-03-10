# Databricks notebook source
# MAGIC %md
# MAGIC # Sample Data Generator
# MAGIC
# MAGIC **Use this notebook if the real EPA CSVs are not yet available.**
# MAGIC Generates ~3 years of realistic synthetic data matching the exact column
# MAGIC schemas of the 4 data sources, including seasonal patterns and simulated
# MAGIC exceedance events for demo purposes.
# MAGIC
# MAGIC Run this notebook INSTEAD of uploading real CSVs, then proceed from
# MAGIC **02_silver_transform** onward.

# COMMAND ----------

CATALOG = "pl_epa_air_quality"
STATION_ID = "PTP01"

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F

np.random.seed(42)

START = datetime(2021, 1, 1)
END   = datetime(2024, 12, 31)

# COMMAND ----------

# MAGIC %md ## Generate Meteorology (10-min)

# COMMAND ----------

met_idx = pd.date_range(START, END, freq="10min")
n = len(met_idx)
t = np.arange(n)

# Seasonal temperature: ~14°C mean, ±10°C seasonal, ±5°C daily
season  = np.sin(2 * np.pi * t / (365.25 * 144))   # 144 = 10-min intervals per day
diurnal = np.sin(2 * np.pi * t / 144 - np.pi / 2)
temp    = 18 + 10 * season + 5 * diurnal + np.random.normal(0, 1.5, n)

# Wind: prevailing NW in winter (210-270°), SW in summer
wind_base_dir = 200 + 40 * season
wind_dir = (wind_base_dir + np.random.normal(0, 30, n)) % 360
wind_spd = np.abs(3 + 2 * season + np.random.exponential(2, n))

# Humidity: inverse of temperature
humidity = np.clip(70 - 15 * season + np.random.normal(0, 8, n), 10, 100)

met_df = pd.DataFrame({
    "Date Time": met_idx.strftime("%d/%m/%Y %H:%M"),
    "Temperature Deg_C": np.round(temp, 1),
    "TSR W/m2": np.clip(
        600 * np.clip(np.sin(2 * np.pi * t / 144), 0, None) + np.random.normal(0, 30, n), 0, None
    ).round(1),
    "Barometric Pressure hPa": np.round(1013 + 8 * season + np.random.normal(0, 3, n), 1),
    "Wind Speed m/s": np.round(wind_spd, 1),
    "Wind Direction deg": np.round(wind_dir, 0),
    "Dew Point Deg_C": np.round(temp - (100 - humidity) / 5, 1),
    "Relative Humidity %": np.round(humidity, 0),
    "Wind Vector E/W m/s": np.round(wind_spd * np.sin(np.radians(wind_dir)), 2),
    "Wind Vector N/S m/s": np.round(wind_spd * np.cos(np.radians(wind_dir)), 2),
})

met_spark = spark.createDataFrame(met_df).withColumn("station_id", F.lit(STATION_ID))
(met_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .option("delta.columnMapping.mode", "name")
 .saveAsTable(f"{CATALOG}.bronze.weather_raw"))
print(f"✓ bronze.weather_raw: {met_spark.count():,} rows (10-min met)")

# COMMAND ----------

# MAGIC %md ## Generate Particle Data (Hourly)

# COMMAND ----------

hrly_idx = pd.date_range(START, END, freq="H")
nh = len(hrly_idx)
th = np.arange(nh)

season_h  = np.sin(2 * np.pi * th / (365.25 * 24))
diurnal_h = np.sin(2 * np.pi * th / 24 - np.pi / 2)

# PM10: higher in dry hot summer months, with occasional dust events
pm10_base = 15 + 8 * season_h + 4 * diurnal_h + np.random.lognormal(0, 0.5, nh) * 3
# Inject ~6 dust event days per year (~18 hours each)
dust_events = np.zeros(nh)
for _ in range(18):                               # 18 events over 3 years
    start_idx = np.random.randint(0, nh - 24)
    dust_events[start_idx:start_idx + 18] += np.random.uniform(60, 180)
pm10 = np.clip(pm10_base + dust_events, 0, None)

pm25 = np.clip(pm10 * 0.45 + np.random.normal(0, 2, nh), 0, None)  # PM2.5 ~45% of PM10

particle_df = pd.DataFrame({
    "Date/Time": hrly_idx.strftime("%d/%m/%Y %H:%M"),
    "PM10 TEOM ug/m3": np.round(pm10, 1),
    "PM2.5 TEOM ug/m3": np.round(pm25, 1),
    "Temperature Deg C": np.round(18 + 10 * season_h + 3 * diurnal_h + np.random.normal(0, 1, nh), 1),
    "Barometric Pressure atm": np.round((1013 + np.random.normal(0, 3, nh)) / 1013.25, 4),
})

part_spark = spark.createDataFrame(particle_df).withColumn("station_id", F.lit(STATION_ID))
(part_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .option("delta.columnMapping.mode", "name")
 .saveAsTable(f"{CATALOG}.bronze.particle_raw"))
print(f"\u2713 bronze.particle_raw: {part_spark.count():,} rows")

# COMMAND ----------

# MAGIC %md ## Generate Gas Data (Hourly)

# COMMAND ----------

# SO2: correlated with smelter operation (weekdays, daytime) + wind
smelter_ops = np.where(
    (hrly_idx.dayofweek < 5) & (hrly_idx.hour.isin(range(6, 22))), 1.0, 0.3
)
wind_factor = np.where(
    (hrly_idx.hour.isin(range(10, 16))), 1.5, 1.0   # afternoon sea breeze carries plume
)
so2 = np.clip(
    0.02 * smelter_ops * wind_factor + np.random.exponential(0.01, nh), 0, None
)
# Inject 4 SO2 exceedance events (> 0.20 ppm)
for _ in range(12):
    idx = np.random.randint(0, nh - 3)
    so2[idx:idx + 3] += np.random.uniform(0.22, 0.45)

gas_df = pd.DataFrame({
    "Date/Time": hrly_idx.strftime("%d/%m/%Y %H:%M"),
    "O3 UVA ppm": np.round(np.clip(0.025 + 0.015 * np.abs(diurnal_h) + np.random.normal(0, 0.005, nh), 0, None), 4),
    "O3 8hr UVA ppm": np.round(np.clip(0.022 + 0.01 * season_h + np.random.normal(0, 0.003, nh), 0, None), 4),
    "NO Chemiluminescence ppm": np.round(np.clip(0.005 * smelter_ops + np.random.exponential(0.002, nh), 0, None), 5),
    "NO2 calc Chemiluminescence ppm": np.round(np.clip(0.01 * smelter_ops + np.random.exponential(0.003, nh), 0, None), 5),
    "NOx Chemiluminescence ppm": np.round(np.clip(0.015 * smelter_ops + np.random.exponential(0.004, nh), 0, None), 5),
    "SO2 UVF ppm": np.round(so2, 5),
    "CO GPC ppm": np.round(np.clip(0.2 + 0.1 * smelter_ops + np.random.normal(0, 0.05, nh), 0.05, None), 3),
    "CO 8 hr GPC ppm": np.round(np.clip(0.18 + 0.08 * smelter_ops + np.random.normal(0, 0.04, nh), 0.05, None), 3),
})

gas_spark = spark.createDataFrame(gas_df).withColumn("station_id", F.lit(STATION_ID))
(gas_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .option("delta.columnMapping.mode", "name")
 .saveAsTable(f"{CATALOG}.bronze.gas_raw"))
print(f"✓ bronze.gas_raw: {gas_spark.count():,} rows")

# COMMAND ----------

# MAGIC %md ## Generate Lead in Air (Daily — Oliver Street)

# COMMAND ----------

daily_idx = pd.date_range(START, END, freq="D")
nd = len(daily_idx)
td = np.arange(nd)

season_d = np.sin(2 * np.pi * td / 365.25)

# Lead: slightly elevated in summer (dry, dusty), baseline ~0.2 ug/m3
# Inject 3 prolonged breach periods (rolling 7d avg > 0.45)
lead_base = 0.20 + 0.08 * season_d + np.random.lognormal(-0.3, 0.4, nd) * 0.1
lead_base = np.clip(lead_base, 0.01, None)

# Breach episodes: elevated daily readings that push rolling avg over threshold
breach_periods = [
    (365 + 30, 25),   # ~Feb 2022, 25 days elevated
    (730 + 180, 20),  # ~Jul 2023, 20 days elevated
    (1000, 15),       # ~Oct 2024, 15 days elevated
]
for start_d, duration in breach_periods:
    if start_d + duration < nd:
        lead_base[start_d:start_d + duration] += np.random.uniform(0.35, 0.65, duration)

lead_df = pd.DataFrame({
    "DateTime": daily_idx.strftime("%-d/%m/%Y"),
    "Location": "Oliver Street",
    "Lead in Air (ug/m3)": np.round(lead_base, 2),
})

lead_spark = spark.createDataFrame(lead_df).withColumn("station_id", F.lit(STATION_ID))
(lead_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .option("delta.columnMapping.mode", "name")
 .saveAsTable(f"{CATALOG}.bronze.lead_in_air_raw"))
print(f"✓ bronze.lead_in_air_raw: {lead_spark.count():,} daily rows (Oliver Street)")

# COMMAND ----------

print("\nSample data generation complete.")
print("Proceed to 02_silver_transform →")