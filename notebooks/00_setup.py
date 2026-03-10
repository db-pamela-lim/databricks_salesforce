# Databricks notebook source
# MAGIC %md
# MAGIC # Port Pirie Air Quality Demo — Setup
# MAGIC ## Joint Demo: Salesforce + Databricks
# MAGIC
# MAGIC This notebook creates the Unity Catalog structure for the demo.
# MAGIC
# MAGIC **Data flow:**
# MAGIC ```
# MAGIC EPA Monitoring CSVs ──► Bronze ──► Silver ──► Gold ──► Salesforce Data360
# MAGIC Salesforce (via Data360) ─────────────────────► Gold (EPA limits join)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 1. Catalog & Schema Setup

# COMMAND ----------

# Configure catalog name — update if your Databricks workspace uses a different catalog
CATALOG = "epa_air_quality"
VOLUME_PATH = f"/Volumes/{CATALOG}/raw/uploads"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

for schema in ["bronze", "silver", "gold", "salesforce"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"✓ Schema {CATALOG}.{schema} ready")

# COMMAND ----------

# MAGIC %md ## 2. Volume for raw CSV uploads

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.uploads
    COMMENT 'Landing zone for EPA monitoring CSV files shared via Salesforce Data360'
""")
print(f"✓ Volume ready at {VOLUME_PATH}")
print()
print("Upload the following CSV files to this volume:")
print("  - Particle data CSVs  → particle/")
print("  - Gas data CSVs       → gas/")
print("  - Met data CSVs       → meteorology/")
print("  - Lead in Air CSVs    → lead_in_air/")

# COMMAND ----------

# MAGIC %md ## 3. EPA Limits Table (Salesforce Data360 → Databricks)
# MAGIC
# MAGIC In the live demo, this table is populated via **Salesforce Data360** sharing the
# MAGIC business license configuration for Nyrstar Port Pirie. The schema mirrors the
# MAGIC Salesforce License & Permit object.
# MAGIC
# MAGIC For demo setup, we seed it here; in production it is written by Data360.

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from datetime import datetime

epa_limits_data = [
    Row(
        license_id="LIC-2024-NPP-001",
        account_name="Nyrstar Port Pirie",
        license_type="Multi-Metal Smelter and Refinery",
        station_id="PTP01",
        pollutant="Lead in Air",
        column_ref="lead_in_air_ug_m3",
        limit_value=0.50,          # EPA NEPM annual standard μg/m³
        site_target=0.45,          # Nyrstar Oliver St site target μg/m³
        unit="ug/m3",
        averaging_period="rolling_7_day",
        source="Salesforce_Data360",
        last_updated=datetime(2024, 7, 1)
    ),
    Row(
        license_id="LIC-2024-NPP-001",
        account_name="Nyrstar Port Pirie",
        license_type="Multi-Metal Smelter and Refinery",
        station_id="PTP01",
        pollutant="PM10",
        column_ref="pm10_teom_ug_m3",
        limit_value=50.0,          # NEPM 24hr standard μg/m³
        site_target=50.0,
        unit="ug/m3",
        averaging_period="24_hour",
        source="Salesforce_Data360",
        last_updated=datetime(2024, 7, 1)
    ),
    Row(
        license_id="LIC-2024-NPP-001",
        account_name="Nyrstar Port Pirie",
        license_type="Multi-Metal Smelter and Refinery",
        station_id="PTP01",
        pollutant="SO2",
        column_ref="so2_uvf_ppm",
        limit_value=0.20,          # NEPM 1hr standard ppm
        site_target=0.20,
        unit="ppm",
        averaging_period="1_hour",
        source="Salesforce_Data360",
        last_updated=datetime(2024, 7, 1)
    ),
]

schema = StructType([
    StructField("license_id", StringType()),
    StructField("account_name", StringType()),
    StructField("license_type", StringType()),
    StructField("station_id", StringType()),
    StructField("pollutant", StringType()),
    StructField("column_ref", StringType()),
    StructField("limit_value", DoubleType()),
    StructField("site_target", DoubleType()),
    StructField("unit", StringType()),
    StructField("averaging_period", StringType()),
    StructField("source", StringType()),
    StructField("last_updated", TimestampType()),
])

epa_limits_df = spark.createDataFrame(epa_limits_data, schema)
(epa_limits_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.salesforce.epa_limits"))

print("✓ salesforce.epa_limits seeded — in live demo this is written by Salesforce Data360")
display(spark.table(f"{CATALOG}.salesforce.epa_limits"))

# COMMAND ----------

# MAGIC %md ## Setup Complete
# MAGIC
# MAGIC Next step: **01_bronze_ingest** — load monitoring CSVs into Bronze Delta tables.