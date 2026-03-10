# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Ingestion
# MAGIC ## EPA Monitoring Data + Nyrstar Lead in Air
# MAGIC
# MAGIC Ingests all 4 data sources from the landing Volume using Auto Loader.
# MAGIC Each table gets a `station_id = "PTP01"` column added, plus ingestion metadata.
# MAGIC
# MAGIC | Source | Grain | Table |
# MAGIC |--------|-------|-------|
# MAGIC | EPA Meteorology | 10-min | `bronze.weather_raw` |
# MAGIC | EPA Particle | Hourly | `bronze.particle_raw` |
# MAGIC | EPA Gaseous | Hourly | `bronze.gas_raw` |
# MAGIC | Nyrstar Lead in Air | Daily | `bronze.lead_in_air_raw` |

# COMMAND ----------

CATALOG = "port_pirie_demo"
STATION_ID = "PTP01"
VOLUME_BASE = f"/Volumes/{CATALOG}/raw/uploads"

# COMMAND ----------

# MAGIC %md ## Helper — common ingestion function

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from datetime import datetime

def ingest_csv_autoloader(source_path, target_table, date_col_raw, date_format,
                           extra_options=None, schema_hints=None):
    """
    Auto Loader ingestion: reads CSVs incrementally, adds station_id and
    ingestion metadata, appends to target Delta table.
    """
    options = {
        "cloudFiles.format": "csv",
        "cloudFiles.schemaLocation": f"{VOLUME_BASE}/_schema/{target_table}",
        "header": "true",
        "inferSchema": "true",
        "encoding": "UTF-8",
    }
    if extra_options:
        options.update(extra_options)

    checkpoint = f"{VOLUME_BASE}/_checkpoints/{target_table}"

    df = (spark.readStream
            .format("cloudFiles")
            .options(**options)
            .load(source_path))

    df = (df
          .withColumn("station_id", F.lit(STATION_ID))
          .withColumn("_source_file", F.col("_metadata.file_path"))
          .withColumn("_ingested_at", F.current_timestamp()))

    # Parse the raw timestamp string to a proper timestamp column
    df = df.withColumn(
        "event_timestamp",
        F.to_timestamp(F.col(date_col_raw), date_format)
    )

    (df.writeStream
       .format("delta")
       .outputMode("append")
       .option("checkpointLocation", checkpoint)
       .option("mergeSchema", "true")
       .trigger(availableNow=True)           # Run-once trigger — perfect for batch demo
       .toTable(f"{CATALOG}.{target_table}")
       .awaitTermination())

    count = spark.table(f"{CATALOG}.{target_table}").count()
    print(f"✓ {CATALOG}.{target_table}: {count:,} rows loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B — Batch read (use if CSVs already present as static files)
# MAGIC
# MAGIC For the live demo with pre-loaded files, the batch path below is simpler to run.
# MAGIC Toggle `USE_AUTOLOADER = True` to switch to incremental Auto Loader mode.

# COMMAND ----------

USE_AUTOLOADER = False  # Set True to demo incremental Auto Loader

def ingest_csv_batch(source_path, target_table, date_col_raw, date_format,
                     delimiter=",", schema_hints=None):
    """
    Batch CSV read: glob all CSVs in source_path, add station_id + metadata.
    Handles multiple CSVs spread across a folder (e.g. one file per year).
    """
    read_opts = {
        "header": "true",
        "inferSchema": "true",
        "encoding": "UTF-8",
        "sep": delimiter,
        "timestampFormat": date_format,
        "pathGlobFilter": "*.csv",
        "recursiveFileLookup": "true",
    }

    df = spark.read.options(**read_opts).csv(source_path)

    df = (df
          .withColumn("station_id", F.lit(STATION_ID))
          .withColumn("_source_file", F.input_file_name())
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("event_timestamp",
                      F.to_timestamp(F.col(date_col_raw), date_format)))

    (df.write
       .format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(f"{CATALOG}.{target_table}"))

    count = spark.table(f"{CATALOG}.{target_table}").count()
    print(f"✓ {CATALOG}.{target_table}: {count:,} rows loaded")

ingest_fn = ingest_csv_autoloader if USE_AUTOLOADER else ingest_csv_batch

# COMMAND ----------

# MAGIC %md ## 1. Meteorology — 10-minute observations
# MAGIC
# MAGIC **Source:** EPA SA — Port Pirie Oliver St Meteorology (station ref: PTP01m10m)
# MAGIC **Columns:** Date Time, Temperature Deg_C, TSR W/m2, Barometric Pressure hPa,
# MAGIC Wind Speed m/s, Wind Direction deg, Dew Point Deg_C, Relative Humidity %,
# MAGIC Wind Vector E/W m/s, Wind Vector N/S m/s

# COMMAND ----------

ingest_fn(
    source_path=f"{VOLUME_BASE}/meteorology/",
    target_table="bronze.weather_raw",
    date_col_raw="Date Time",
    date_format="dd/MM/yyyy HH:mm",
)

# COMMAND ----------

# MAGIC %md ## 2. Particle Data — Hourly
# MAGIC
# MAGIC **Source:** EPA SA — Port Pirie Oliver St Particle (station ref: PTP01p)
# MAGIC **Columns:** Date/Time, PM10 TEOM ug/m3, PM2.5 TEOM ug/m3,
# MAGIC Temperature Deg C, Barometric Pressure atm

# COMMAND ----------

ingest_fn(
    source_path=f"{VOLUME_BASE}/particle/",
    target_table="bronze.particle_raw",
    date_col_raw="Date/Time",
    date_format="dd/MM/yyyy HH:mm",
)

# COMMAND ----------

# MAGIC %md ## 3. Gaseous Data — Hourly
# MAGIC
# MAGIC **Source:** EPA SA — Port Pirie Oliver St Gaseous (station ref: PTP01g)
# MAGIC **Columns:** Date/Time, O3 UVA ppm, O3 8hr UVA ppm, NO Chemiluminescence ppm,
# MAGIC NO2 calc Chemiluminescence ppm, NOx Chemiluminescence ppm,
# MAGIC SO2 UVF ppm, CO GPC ppm, CO 8 hr GPC ppm

# COMMAND ----------

ingest_fn(
    source_path=f"{VOLUME_BASE}/gas/",
    target_table="bronze.gas_raw",
    date_col_raw="Date/Time",
    date_format="dd/MM/yyyy HH:mm",
)

# COMMAND ----------

# MAGIC %md ## 4. Lead in Air — Daily (Oliver Street only)
# MAGIC
# MAGIC **Source:** Nyrstar / portpirieairquality.com.au
# MAGIC **Filter:** Location = "Oliver Street" (file may contain multiple sites)

# COMMAND ----------

ingest_fn(
    source_path=f"{VOLUME_BASE}/lead_in_air/",
    target_table="bronze.lead_in_air_raw",
    date_col_raw="DateTime",
    date_format="d/MM/yyyy",
)

# Filter to Oliver Street only — other locations not relevant to PTP01
spark.sql(f"""
    DELETE FROM {CATALOG}.bronze.lead_in_air_raw
    WHERE lower(Location) != 'oliver street'
      AND Location IS NOT NULL
""")
print("✓ Filtered to Oliver Street observations only")

# COMMAND ----------

# MAGIC %md ## Bronze Summary

# COMMAND ----------

tables = ["bronze.weather_raw", "bronze.particle_raw", "bronze.gas_raw", "bronze.lead_in_air_raw"]
for t in tables:
    df = spark.table(f"{CATALOG}.{t}")
    print(f"{t}: {df.count():,} rows | {len(df.columns)} columns")
    print(f"  station_id values: {[r[0] for r in df.select('station_id').distinct().collect()]}")
    print()
