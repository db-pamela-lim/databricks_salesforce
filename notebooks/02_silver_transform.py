# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Cleanse & Standardise
# MAGIC
# MAGIC - Standardise column names (snake_case, no special characters)
# MAGIC - Cast to correct data types; flag nulls from sensor outages
# MAGIC - Resample meteorology from 10-min to 1-hour to align with other sources
# MAGIC - Produce `silver.combined_hourly` — single joined table for ML & analytics

# COMMAND ----------

CATALOG = "port_pirie_demo"

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md ## 1. Silver Meteorology (10-min → 1-hour)

# COMMAND ----------

weather_raw = spark.table(f"{CATALOG}.bronze.weather_raw")

weather_silver = (weather_raw
    .select(
        F.col("station_id"),
        F.col("event_timestamp"),
        F.col("`Temperature Deg_C`").cast("double").alias("temperature_c"),
        F.col("`TSR W/m2`").cast("double").alias("solar_radiation_w_m2"),
        F.col("`Barometric Pressure hPa`").cast("double").alias("barometric_pressure_hpa"),
        F.col("`Wind Speed m/s`").cast("double").alias("wind_speed_m_s"),
        F.col("`Wind Direction deg`").cast("double").alias("wind_direction_deg"),
        F.col("`Dew Point Deg_C`").cast("double").alias("dew_point_c"),
        F.col("`Relative Humidity %`").cast("double").alias("relative_humidity_pct"),
        F.col("`Wind Vector E/W m/s`").cast("double").alias("wind_vector_ew_m_s"),
        F.col("`Wind Vector N/S m/s`").cast("double").alias("wind_vector_ns_m_s"),
    )
    .withColumn("data_quality_flag",
        F.when(F.col("temperature_c").isNull() | F.col("wind_speed_m_s").isNull(), "SENSOR_OUTAGE")
         .otherwise("OK"))
)

# Resample: aggregate 10-min readings to 1-hour windows
# Wind direction: vector mean via E/W and N/S components
weather_hourly = (weather_silver
    .withColumn("hour_ts", F.date_trunc("hour", F.col("event_timestamp")))
    .groupBy("station_id", "hour_ts")
    .agg(
        F.avg("temperature_c").alias("temperature_c"),
        F.avg("solar_radiation_w_m2").alias("solar_radiation_w_m2"),
        F.avg("barometric_pressure_hpa").alias("barometric_pressure_hpa"),
        F.max("wind_speed_m_s").alias("wind_speed_max_m_s"),   # max gust in hour
        F.avg("wind_speed_m_s").alias("wind_speed_avg_m_s"),
        F.avg("wind_vector_ew_m_s").alias("wind_vector_ew_m_s"),  # vector mean
        F.avg("wind_vector_ns_m_s").alias("wind_vector_ns_m_s"),
        # Derived vector mean direction
        F.degrees(F.atan2(
            F.avg("wind_vector_ew_m_s"),
            F.avg("wind_vector_ns_m_s")
        )).alias("wind_direction_vector_mean_deg"),
        F.avg("dew_point_c").alias("dew_point_c"),
        F.avg("relative_humidity_pct").alias("relative_humidity_pct"),
        F.count("*").alias("obs_count_per_hour"),   # expect 6; fewer = sensor gaps
    )
    .withColumn("data_quality_flag",
        F.when(F.col("obs_count_per_hour") < 4, "INCOMPLETE_HOUR")
         .otherwise("OK"))
)

(weather_hourly.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.silver.weather"))

print(f"✓ silver.weather: {weather_hourly.count():,} hourly records")

# COMMAND ----------

# MAGIC %md ## 2. Silver Particle Data

# COMMAND ----------

particle_raw = spark.table(f"{CATALOG}.bronze.particle_raw")

particle_silver = (particle_raw
    .select(
        F.col("station_id"),
        F.col("event_timestamp"),
        F.col("`PM10 TEOM ug/m3`").cast("double").alias("pm10_teom_ug_m3"),
        F.col("`PM2.5 TEOM ug/m3`").cast("double").alias("pm25_teom_ug_m3"),
        F.col("`Temperature Deg C`").cast("double").alias("temperature_c"),
        F.col("`Barometric Pressure atm`").cast("double").alias("barometric_pressure_atm"),
    )
    .withColumn("data_quality_flag",
        F.when(F.col("pm10_teom_ug_m3") < 0, "NEGATIVE_VALUE")
         .when(F.col("pm10_teom_ug_m3") > 1000, "SPIKE")
         .when(F.col("pm10_teom_ug_m3").isNull(), "MISSING")
         .otherwise("OK"))
)

(particle_silver.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.silver.particle"))

print(f"✓ silver.particle: {particle_silver.count():,} rows")

# COMMAND ----------

# MAGIC %md ## 3. Silver Gas Data

# COMMAND ----------

gas_raw = spark.table(f"{CATALOG}.bronze.gas_raw")

gas_silver = (gas_raw
    .select(
        F.col("station_id"),
        F.col("event_timestamp"),
        F.col("`O3 UVA ppm`").cast("double").alias("o3_uva_ppm"),
        F.col("`O3 8hr UVA ppm`").cast("double").alias("o3_8hr_uva_ppm"),
        F.col("`NO Chemiluminescence ppm`").cast("double").alias("no_ppm"),
        F.col("`NO2 calc Chemiluminescence ppm`").cast("double").alias("no2_ppm"),
        F.col("`NOx Chemiluminescence ppm`").cast("double").alias("nox_ppm"),
        F.col("`SO2 UVF ppm`").cast("double").alias("so2_uvf_ppm"),
        F.col("`CO GPC ppm`").cast("double").alias("co_ppm"),
        F.col("`CO 8 hr GPC ppm`").cast("double").alias("co_8hr_ppm"),
    )
    .withColumn("data_quality_flag",
        F.when(F.col("so2_uvf_ppm") < 0, "NEGATIVE_VALUE")
         .when(F.col("so2_uvf_ppm") > 10, "SPIKE")
         .when(F.col("so2_uvf_ppm").isNull(), "MISSING")
         .otherwise("OK"))
)

(gas_silver.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.silver.gas"))

print(f"✓ silver.gas: {gas_silver.count():,} rows")

# COMMAND ----------

# MAGIC %md ## 4. Silver Lead in Air (Daily)

# COMMAND ----------

lead_raw = spark.table(f"{CATALOG}.bronze.lead_in_air_raw")

lead_silver = (lead_raw
    .select(
        F.col("station_id"),
        F.to_date(F.col("event_timestamp")).alias("measurement_date"),
        F.col("`Lead in Air (ug/m3)`").cast("double").alias("lead_in_air_ug_m3"),
    )
    .withColumn("data_quality_flag",
        F.when(F.col("lead_in_air_ug_m3") < 0, "NEGATIVE_VALUE")
         .when(F.col("lead_in_air_ug_m3") > 10, "SPIKE")
         .when(F.col("lead_in_air_ug_m3").isNull(), "MISSING")
         .otherwise("OK"))
    .dropDuplicates(["station_id", "measurement_date"])
)

(lead_silver.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.silver.lead_in_air"))

print(f"✓ silver.lead_in_air: {lead_silver.count():,} daily observations")

# COMMAND ----------

# MAGIC %md ## 5. Combined Hourly Table
# MAGIC
# MAGIC Joins particle + gas + weather on station_id and hour timestamp.
# MAGIC Lead in Air is a daily grain so we forward-fill to each hour (labelled "daily_obs").

# COMMAND ----------

particle = spark.table(f"{CATALOG}.silver.particle")
gas      = spark.table(f"{CATALOG}.silver.gas")
weather  = spark.table(f"{CATALOG}.silver.weather")
lead     = spark.table(f"{CATALOG}.silver.lead_in_air")

# Forward-fill daily lead value across all hours of that day
lead_hourly = (lead
    .withColumn("hour_ts",
        F.expr("explode(sequence(measurement_date, measurement_date, interval 1 hour))"))
    .withColumn("hour_ts", F.col("hour_ts").cast("timestamp"))
    .select("station_id", "hour_ts", "lead_in_air_ug_m3")
)

combined = (particle
    .join(gas,     ["station_id", "event_timestamp"], "outer")
    .join(weather, particle["station_id"] == weather["station_id"], "left")
        .where(particle["event_timestamp"] == weather["hour_ts"])
        .drop(weather["station_id"])
    .join(lead_hourly,
          (particle["station_id"] == lead_hourly["station_id"]) &
          (F.date_trunc("hour", particle["event_timestamp"]) == lead_hourly["hour_ts"]),
          "left")
    .drop(lead_hourly["station_id"], lead_hourly["hour_ts"])
    .select(
        particle["station_id"],
        particle["event_timestamp"],
        "pm10_teom_ug_m3", "pm25_teom_ug_m3",
        "so2_uvf_ppm", "o3_uva_ppm", "o3_8hr_uva_ppm",
        "no_ppm", "no2_ppm", "nox_ppm", "co_ppm", "co_8hr_ppm",
        "temperature_c", "barometric_pressure_hpa",
        "wind_speed_avg_m_s", "wind_speed_max_m_s",
        "wind_direction_vector_mean_deg",
        "relative_humidity_pct", "dew_point_c",
        "solar_radiation_w_m2",
        "lead_in_air_ug_m3",
    )
)

(combined.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.silver.combined_hourly"))

print(f"✓ silver.combined_hourly: {combined.count():,} rows")
display(spark.table(f"{CATALOG}.silver.combined_hourly").limit(5))
