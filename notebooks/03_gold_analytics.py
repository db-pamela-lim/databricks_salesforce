# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Analytics & Permit Status
# MAGIC
# MAGIC Produces the views consumed by:
# MAGIC - **Databricks SQL Dashboard** (live visualisations)
# MAGIC - **Salesforce Data360** (data activations — investigation case trigger)
# MAGIC
# MAGIC | Gold Table / View | Purpose |
# MAGIC |---|---|
# MAGIC | `gold.lead_rolling_weekly_avg` | 7-day rolling average — **Salesforce activation source** |
# MAGIC | `gold.permit_status_current` | Live permit gate per pollutant vs EPA limits |
# MAGIC | `gold.exceedance_log` | Audit trail of every threshold breach |
# MAGIC | `gold.pollutant_daily_summary` | Daily aggregates for dashboard time-series |
# MAGIC | `gold.pollutant_monthly_percentiles` | P50/P95/Max per month for trend charts |

# COMMAND ----------

CATALOG = "epa_air_quality"

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md ## 1. Lead in Air — 7-Day Rolling Average
# MAGIC
# MAGIC **This view is the Salesforce Data360 activation source.**
# MAGIC When `rolling_7day_avg_lead_ug_m3 > site_target`, Data360 automatically
# MAGIC creates an investigation case on the Nyrstar Port Pirie account.

# COMMAND ----------

# Seed the EPA regulatory limits table (normally synced from Salesforce)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.salesforce.epa_limits (
        station_id       STRING,
        pollutant        STRING,
        limit_value      DOUBLE  COMMENT 'SA EPA regulatory limit',
        site_target      DOUBLE  COMMENT 'Nyrstar site-specific target (stricter)',
        unit             STRING,
        averaging_period STRING
    )
""")
spark.sql(f"""
    INSERT OVERWRITE {CATALOG}.salesforce.epa_limits
    VALUES
        ('PTP01', 'Lead in Air', 0.50, 0.45, 'ug/m3', '7-day rolling'),
        ('PTP01', 'PM10',        50.0, 40.0, 'ug/m3', '24-hour'),
        ('PTP01', 'SO2',         0.20, 0.15, 'ppm',   '1-hour')
""")

lead = spark.table(f"{CATALOG}.silver.lead_in_air")
limits = spark.table(f"{CATALOG}.salesforce.epa_limits").filter("pollutant = 'Lead in Air'")

# 7-day rolling window (current day + 6 prior)
w7 = Window.partitionBy("station_id").orderBy("measurement_date").rowsBetween(-6, 0)

lead_rolling = (lead
    .filter("data_quality_flag = 'OK'")
    .withColumn("rolling_7day_avg_lead_ug_m3",
        F.round(F.avg("lead_in_air_ug_m3").over(w7), 3))
    .withColumn("rolling_7day_days_in_window",
        F.count("lead_in_air_ug_m3").over(w7))   # <7 = window not yet full at start
)

# Join EPA limits from Salesforce
lead_rolling = (lead_rolling
    .join(limits.select("station_id", "limit_value", "site_target"),
          "station_id", "left")
    .withColumn("pct_of_epa_limit",
        F.round(F.col("rolling_7day_avg_lead_ug_m3") / F.col("limit_value") * 100, 1))
    .withColumn("pct_of_site_target",
        F.round(F.col("rolling_7day_avg_lead_ug_m3") / F.col("site_target") * 100, 1))
    .withColumn("compliance_status",
        F.when(F.col("rolling_7day_avg_lead_ug_m3") > F.col("limit_value"), "EPA_BREACH")
         .when(F.col("rolling_7day_avg_lead_ug_m3") > F.col("site_target"), "SITE_TARGET_BREACH")
         .when(F.col("rolling_7day_avg_lead_ug_m3") > F.col("site_target") * 0.85, "APPROACHING_TARGET")
         .otherwise("COMPLIANT"))
    # Flag for Salesforce Data360 activation — boolean for clean trigger condition
    .withColumn("trigger_investigation",
        F.when(F.col("rolling_7day_avg_lead_ug_m3") > F.col("site_target"), True)
         .otherwise(False))
)

(lead_rolling.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.gold.lead_rolling_weekly_avg"))

# Add table comment for Genie AI/BI
spark.sql(f"""
    COMMENT ON TABLE {CATALOG}.gold.lead_rolling_weekly_avg IS
    '7-day rolling average of Lead in Air (ug/m3) at Port Pirie Oliver Street (PTP01).
     trigger_investigation=true when rolling average exceeds the Nyrstar site target of 0.45 ug/m3.
     This table is shared to Salesforce Data360 to automatically create investigation cases.'
""")

print(f"✓ gold.lead_rolling_weekly_avg: {lead_rolling.count():,} daily records")

# Preview recent breach periods
breach_preview = (spark.table(f"{CATALOG}.gold.lead_rolling_weekly_avg")
    .filter("compliance_status != 'COMPLIANT'")
    .orderBy("measurement_date", ascending=False)
    .limit(10))
display(breach_preview)

# COMMAND ----------

# MAGIC %md ## 2. Current Permit Status (all three pollutants)
# MAGIC
# MAGIC Compares the **most recent valid reading** for PM10, SO2, and Lead against
# MAGIC the EPA limits from the Salesforce business license.

# COMMAND ----------

combined = spark.table(f"{CATALOG}.silver.combined_hourly")
lead_gold = spark.table(f"{CATALOG}.gold.lead_rolling_weekly_avg")
limits_all = spark.table(f"{CATALOG}.salesforce.epa_limits")

# Latest hourly reading
latest_hour = (combined
    .filter("pm10_teom_ug_m3 IS NOT NULL OR so2_uvf_ppm IS NOT NULL")
    .orderBy("event_timestamp", ascending=False)
    .limit(1)
)

# Latest lead rolling average
latest_lead = (lead_gold
    .orderBy("measurement_date", ascending=False)
    .limit(1)
    .select("station_id", "measurement_date",
            "rolling_7day_avg_lead_ug_m3", "site_target",
            "compliance_status", "trigger_investigation",
            "pct_of_site_target")
)

# Build permit status rows per pollutant
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, TimestampType

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.gold.permit_status_current AS
    WITH latest_readings AS (
        SELECT * FROM (
            SELECT
                station_id,
                event_timestamp,
                pm10_teom_ug_m3   AS current_value,
                'PM10'             AS pollutant,
                'ug/m3'            AS unit
            FROM {CATALOG}.silver.combined_hourly
            WHERE pm10_teom_ug_m3 IS NOT NULL
            ORDER BY event_timestamp DESC LIMIT 1
        )

        UNION ALL

        SELECT * FROM (
            SELECT
                station_id,
                event_timestamp,
                so2_uvf_ppm        AS current_value,
                'SO2'              AS pollutant,
                'ppm'              AS unit
            FROM {CATALOG}.silver.combined_hourly
            WHERE so2_uvf_ppm IS NOT NULL
            ORDER BY event_timestamp DESC LIMIT 1
        )

        UNION ALL

        SELECT * FROM (
            SELECT
                station_id,
                CAST(measurement_date AS TIMESTAMP) AS event_timestamp,
                rolling_7day_avg_lead_ug_m3         AS current_value,
                'Lead in Air'                        AS pollutant,
                'ug/m3'                              AS unit
            FROM {CATALOG}.gold.lead_rolling_weekly_avg
            ORDER BY measurement_date DESC LIMIT 1
        )
    )
    SELECT
        r.station_id,
        r.event_timestamp  AS reading_timestamp,
        r.pollutant,
        r.current_value,
        r.unit,
        l.limit_value      AS epa_limit,
        l.site_target,
        l.averaging_period,
        ROUND(r.current_value / l.site_target * 100, 1) AS pct_of_site_target,
        CASE
            WHEN r.current_value > l.limit_value THEN 'EPA_BREACH'
            WHEN r.current_value > l.site_target THEN 'SITE_TARGET_BREACH'
            WHEN r.current_value > l.site_target * 0.85 THEN 'APPROACHING'
            ELSE 'COMPLIANT'
        END AS permit_status,
        CASE
            WHEN r.current_value > l.site_target THEN true
            ELSE false
        END AS action_required,
        current_timestamp() AS assessed_at
    FROM latest_readings r
    LEFT JOIN {CATALOG}.salesforce.epa_limits l
      ON r.station_id = l.station_id
     AND r.pollutant  = l.pollutant
""")

print("\u2713 gold.permit_status_current")
display(spark.table(f"{CATALOG}.gold.permit_status_current"))

# COMMAND ----------

# MAGIC %md ## 3. Exceedance Log (Audit Trail)
# MAGIC
# MAGIC Every hourly observation where any monitored pollutant exceeded its site target.
# MAGIC Immutable append-only history — feeds the "compliance history" chart.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.gold.exceedance_log AS
    WITH pm10_breaches AS (
        SELECT
            h.station_id,
            h.event_timestamp,
            'PM10'                AS pollutant,
            h.pm10_teom_ug_m3     AS observed_value,
            l.site_target         AS threshold,
            l.unit,
            ROUND(h.pm10_teom_ug_m3 / l.site_target * 100, 1) AS pct_of_threshold
        FROM {CATALOG}.silver.combined_hourly h
        JOIN {CATALOG}.salesforce.epa_limits l
          ON h.station_id = l.station_id AND l.pollutant = 'PM10'
        WHERE h.pm10_teom_ug_m3 > l.site_target
    ),
    so2_breaches AS (
        SELECT
            h.station_id,
            h.event_timestamp,
            'SO2'                 AS pollutant,
            h.so2_uvf_ppm         AS observed_value,
            l.site_target         AS threshold,
            l.unit,
            ROUND(h.so2_uvf_ppm / l.site_target * 100, 1) AS pct_of_threshold
        FROM {CATALOG}.silver.combined_hourly h
        JOIN {CATALOG}.salesforce.epa_limits l
          ON h.station_id = l.station_id AND l.pollutant = 'SO2'
        WHERE h.so2_uvf_ppm > l.site_target
    ),
    lead_breaches AS (
        SELECT
            station_id,
            CAST(measurement_date AS TIMESTAMP) AS event_timestamp,
            'Lead in Air'                        AS pollutant,
            rolling_7day_avg_lead_ug_m3          AS observed_value,
            site_target                          AS threshold,
            'ug/m3'                              AS unit,
            pct_of_site_target                   AS pct_of_threshold
        FROM {CATALOG}.gold.lead_rolling_weekly_avg
        WHERE trigger_investigation = true
    )
    SELECT *, current_timestamp() AS logged_at
    FROM pm10_breaches
    UNION ALL SELECT *, current_timestamp() FROM so2_breaches
    UNION ALL SELECT *, current_timestamp() FROM lead_breaches
    ORDER BY event_timestamp DESC
""")

exceedance_count = spark.table(f"{CATALOG}.gold.exceedance_log").count()
print(f"✓ gold.exceedance_log: {exceedance_count:,} threshold exceedance events recorded")

# COMMAND ----------

# MAGIC %md ## 4. Daily Summary (Dashboard Time-Series)

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.gold.pollutant_daily_summary AS
    SELECT
        h.station_id,
        DATE(h.event_timestamp)          AS measurement_date,
        -- PM10
        ROUND(AVG(h.pm10_teom_ug_m3), 2)    AS pm10_avg_ug_m3,
        ROUND(MAX(h.pm10_teom_ug_m3), 2)    AS pm10_max_ug_m3,
        -- SO2
        ROUND(AVG(h.so2_uvf_ppm), 4)        AS so2_avg_ppm,
        ROUND(MAX(h.so2_uvf_ppm), 4)        AS so2_max_ppm,
        -- Lead (daily obs joined from silver)
        ROUND(AVG(l.lead_in_air_ug_m3), 3)  AS lead_daily_obs_ug_m3,
        ROUND(AVG(rw.rolling_7day_avg_lead_ug_m3), 3) AS lead_rolling_7day_avg,
        -- EPA limits for reference lines on charts
        0.50                                 AS lead_epa_limit,
        0.45                                 AS lead_site_target,
        50.0                                 AS pm10_epa_limit,
        0.20                                 AS so2_epa_limit,
        -- Met
        ROUND(AVG(h.wind_speed_avg_m_s), 2) AS wind_speed_avg,
        ROUND(AVG(h.wind_direction_vector_mean_deg), 1) AS wind_direction_avg,
        ROUND(AVG(h.temperature_c), 1)       AS temperature_avg_c,
        -- Data completeness
        COUNT(h.pm10_teom_ug_m3)             AS pm10_obs_count,  -- expect 24
        COUNT(h.so2_uvf_ppm)                 AS so2_obs_count
    FROM {CATALOG}.silver.combined_hourly h
    LEFT JOIN {CATALOG}.silver.lead_in_air l
      ON h.station_id = l.station_id
     AND DATE(h.event_timestamp) = l.measurement_date
    LEFT JOIN {CATALOG}.gold.lead_rolling_weekly_avg rw
      ON h.station_id = rw.station_id
     AND DATE(h.event_timestamp) = rw.measurement_date
    GROUP BY h.station_id, DATE(h.event_timestamp)
    ORDER BY measurement_date
""")

print(f"✓ gold.pollutant_daily_summary: {spark.table(f'{CATALOG}.gold.pollutant_daily_summary').count():,} daily rows")

# COMMAND ----------

# MAGIC %md ## 5. Monthly Percentiles (Trend Overview Chart)

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.gold.pollutant_monthly_stats AS
    SELECT
        station_id,
        DATE_TRUNC('month', measurement_date) AS month,
        -- Lead in Air monthly stats
        ROUND(AVG(lead_daily_obs_ug_m3), 3)                        AS lead_monthly_avg,
        ROUND(PERCENTILE(lead_daily_obs_ug_m3, 0.50), 3)           AS lead_p50,
        ROUND(PERCENTILE(lead_daily_obs_ug_m3, 0.95), 3)           AS lead_p95,
        ROUND(MAX(lead_daily_obs_ug_m3), 3)                        AS lead_max,
        SUM(CASE WHEN lead_rolling_7day_avg > 0.45 THEN 1 ELSE 0 END) AS days_site_target_breached,
        SUM(CASE WHEN lead_rolling_7day_avg > 0.50 THEN 1 ELSE 0 END) AS days_epa_limit_breached,
        -- PM10
        ROUND(AVG(pm10_avg_ug_m3), 1)                              AS pm10_monthly_avg,
        ROUND(MAX(pm10_max_ug_m3), 1)                              AS pm10_monthly_max,
        -- SO2
        ROUND(AVG(so2_avg_ppm), 4)                                 AS so2_monthly_avg,
        ROUND(MAX(so2_max_ppm), 4)                                 AS so2_monthly_max
    FROM {CATALOG}.gold.pollutant_daily_summary
    GROUP BY station_id, DATE_TRUNC('month', measurement_date)
    ORDER BY month
""")

print(f"✓ gold.pollutant_monthly_stats ready")
display(spark.table(f"{CATALOG}.gold.pollutant_monthly_stats").limit(12))

# COMMAND ----------

# MAGIC %md ## Gold Layer Summary
# MAGIC
# MAGIC | Table | Rows | Consumer |
# MAGIC |-------|------|----------|
# MAGIC | `gold.lead_rolling_weekly_avg` | — | **Salesforce Data360 activation** |
# MAGIC | `gold.permit_status_current` | 3 (one per pollutant) | Dashboard — status card |
# MAGIC | `gold.exceedance_log` | — | Dashboard — breach history |
# MAGIC | `gold.pollutant_daily_summary` | — | Dashboard — time-series charts |
# MAGIC | `gold.pollutant_monthly_stats` | — | Dashboard — trend overview |
# MAGIC
# MAGIC Next step: **04_ml_exceedance_prediction** → predict lead breaches 48 hrs ahead.