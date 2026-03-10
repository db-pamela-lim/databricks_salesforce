-- =============================================================================
-- Port Pirie Air Quality — Databricks SQL Dashboard Queries
-- Salesforce + Databricks Joint Demo
-- =============================================================================
-- These queries back the Databricks SQL dashboard widgets.
-- Parameter: :station_id (default PTP01), :date_from, :date_to
-- =============================================================================


-- -----------------------------------------------------------------------------
-- WIDGET 1: Permit Status Cards (3 tiles — one per pollutant)
-- Shows current compliance status fed from the Salesforce business license
-- -----------------------------------------------------------------------------
SELECT
    pollutant,
    ROUND(current_value, 3)          AS current_value,
    unit,
    epa_limit,
    site_target,
    pct_of_site_target,
    permit_status,
    action_required,
    averaging_period,
    DATE_FORMAT(reading_timestamp, 'dd MMM yyyy HH:mm') AS as_at
FROM pl_epa_air_quality.gold.permit_status_current
WHERE station_id = :station_id
ORDER BY
    CASE permit_status
        WHEN 'EPA_BREACH'         THEN 1
        WHEN 'SITE_TARGET_BREACH' THEN 2
        WHEN 'APPROACHING'        THEN 3
        ELSE 4
    END;


-- -----------------------------------------------------------------------------
-- WIDGET 2: Lead in Air — 7-Day Rolling Average vs Thresholds (Line chart)
-- KEY CHART: This is the Salesforce Data360 activation trigger source
-- -----------------------------------------------------------------------------
SELECT
    measurement_date,
    lead_in_air_ug_m3                           AS daily_observation,
    rolling_7day_avg_lead_ug_m3                 AS rolling_7day_avg,
    0.45                                        AS site_target_threshold,
    0.50                                        AS epa_limit,
    compliance_status,
    trigger_investigation
FROM pl_epa_air_quality.gold.lead_rolling_weekly_avg
WHERE station_id  = :station_id
  AND measurement_date BETWEEN :date_from AND :date_to
ORDER BY measurement_date;


-- -----------------------------------------------------------------------------
-- WIDGET 3: Lead — Monthly Performance vs Target (Bar + line combo)
-- Shows month-by-month average with target line and breach day count
-- -----------------------------------------------------------------------------
SELECT
    DATE_FORMAT(month, 'MMM yyyy')              AS month_label,
    month,
    ROUND(lead_monthly_avg, 3)                  AS monthly_avg_lead_ug_m3,
    ROUND(lead_p50, 3)                          AS median_lead_ug_m3,
    ROUND(lead_p95, 3)                          AS p95_lead_ug_m3,
    ROUND(lead_max, 3)                          AS max_lead_ug_m3,
    days_site_target_breached,
    days_epa_limit_breached,
    0.45                                        AS site_target,
    0.50                                        AS epa_limit
FROM pl_epa_air_quality.gold.pollutant_monthly_stats
WHERE station_id  = :station_id
  AND month BETWEEN DATE_TRUNC('month', CAST(:date_from AS DATE))
                AND DATE_TRUNC('month', CAST(:date_to   AS DATE))
ORDER BY month;


-- -----------------------------------------------------------------------------
-- WIDGET 4: PM10 Daily Average vs EPA Limit (Area chart)
-- -----------------------------------------------------------------------------
SELECT
    measurement_date,
    ROUND(pm10_avg_ug_m3, 1)    AS pm10_daily_avg_ug_m3,
    ROUND(pm10_max_ug_m3, 1)    AS pm10_daily_max_ug_m3,
    50.0                        AS epa_limit_ug_m3
FROM pl_epa_air_quality.gold.pollutant_daily_summary
WHERE station_id     = :station_id
  AND measurement_date BETWEEN :date_from AND :date_to
ORDER BY measurement_date;


-- -----------------------------------------------------------------------------
-- WIDGET 5: SO2 Hourly Peak vs EPA Limit (Scatter / line chart)
-- -----------------------------------------------------------------------------
SELECT
    measurement_date,
    ROUND(so2_avg_ppm, 4)       AS so2_daily_avg_ppm,
    ROUND(so2_max_ppm, 4)       AS so2_daily_max_ppm,
    0.20                        AS epa_limit_ppm
FROM pl_epa_air_quality.gold.pollutant_daily_summary
WHERE station_id     = :station_id
  AND measurement_date BETWEEN :date_from AND :date_to
ORDER BY measurement_date;


-- -----------------------------------------------------------------------------
-- WIDGET 6: ML Breach Prediction — Risk Band (for next 7 days)
-- -----------------------------------------------------------------------------
SELECT
    measurement_date,
    ROUND(lead_rolling_7day_avg, 3)             AS rolling_avg,
    ROUND(breach_probability_3d * 100, 1)       AS breach_probability_pct,
    risk_band,
    actual_breach,
    breach_predicted_3d
FROM pl_epa_air_quality.gold.breach_predictions
WHERE measurement_date BETWEEN :date_from AND :date_to
ORDER BY measurement_date;


-- -----------------------------------------------------------------------------
-- WIDGET 7: Exceedance Event Log (Table widget — drilldown)
-- -----------------------------------------------------------------------------
SELECT
    DATE_FORMAT(event_timestamp, 'dd MMM yyyy HH:mm')  AS event_time,
    pollutant,
    ROUND(observed_value, 4)                            AS observed_value,
    unit,
    ROUND(threshold, 4)                                 AS site_target,
    pct_of_threshold                                    AS pct_of_target,
    CASE
        WHEN pct_of_threshold > 150 THEN '🔴 CRITICAL'
        WHEN pct_of_threshold > 110 THEN '🟠 BREACH'
        ELSE '🟡 APPROACHING'
    END                                                 AS severity
FROM pl_epa_air_quality.gold.exceedance_log
WHERE event_timestamp BETWEEN :date_from AND :date_to
ORDER BY event_timestamp DESC
LIMIT 100;


-- -----------------------------------------------------------------------------
-- WIDGET 8: Wind Rose data — what wind conditions precede high lead days?
-- (For scatter plot: wind direction vs lead concentration)
-- -----------------------------------------------------------------------------
SELECT
    d.measurement_date,
    ROUND(d.wind_direction_avg, 0)              AS wind_direction_deg,
    ROUND(d.wind_speed_avg, 1)                  AS wind_speed_m_s,
    ROUND(d.lead_daily_obs_ug_m3, 3)            AS lead_ug_m3,
    CASE
        WHEN d.lead_daily_obs_ug_m3 > 0.45 THEN 'Above target'
        WHEN d.lead_daily_obs_ug_m3 > 0.35 THEN 'Approaching'
        ELSE 'Compliant'
    END                                         AS lead_status,
    p.risk_band                                 AS ml_risk_band
FROM pl_epa_air_quality.gold.pollutant_daily_summary d
LEFT JOIN pl_epa_air_quality.gold.breach_predictions p
  ON d.measurement_date = p.measurement_date
WHERE d.station_id     = :station_id
  AND d.measurement_date BETWEEN :date_from AND :date_to
  AND d.wind_direction_avg IS NOT NULL
ORDER BY d.measurement_date;


-- -----------------------------------------------------------------------------
-- WIDGET 9: Salesforce License Summary (header context card)
-- Shows the business license data shared from Salesforce via Data360
-- -----------------------------------------------------------------------------
SELECT
    account_name                                AS account,
    license_type,
    station_id                                  AS monitoring_station,
    pollutant,
    ROUND(limit_value, 3)                       AS epa_limit,
    ROUND(site_target, 3)                       AS site_target,
    unit,
    averaging_period,
    source                                      AS data_source
FROM pl_epa_air_quality.salesforce.epa_limits
WHERE station_id = :station_id
ORDER BY pollutant;
