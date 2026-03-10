# Databricks notebook source
# MAGIC %md
# MAGIC # ML — Lead Exceedance Prediction
# MAGIC
# MAGIC **Goal:** Predict whether the 7-day rolling average of Lead in Air will breach
# MAGIC the 0.45 μg/m³ site target in the next **3 days**, using meteorological and
# MAGIC recent air quality readings as features.
# MAGIC
# MAGIC **Why this matters for permits:**
# MAGIC A 3-day ahead warning lets safety officers pre-condition work permits and
# MAGIC schedule high-dust outdoor maintenance during forecast low-risk windows.
# MAGIC
# MAGIC **Approach:**
# MAGIC - Binary classification: `breach_in_3_days` = 1 / 0
# MAGIC - Features: wind speed/direction, temperature, humidity, recent PM10, SO2
# MAGIC - Algorithm: XGBoost (via MLlib) + MLflow tracking
# MAGIC - Output: `gold.breach_predictions` — probability score per day

# COMMAND ----------

# DBTITLE 1,Fix pydantic dependency
# MAGIC %pip install mlflow "pydantic<2" -q

# COMMAND ----------

CATALOG = "epa_air_quality"
EXPERIMENT_NAME = "/Users/pamela.lim@databricks.com/port_pirie_lead_exceedance"
MODEL_NAME = "port_pirie_lead_breach_predictor"

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md ## 1. Feature Engineering

# COMMAND ----------

daily = spark.table(f"{CATALOG}.gold.pollutant_daily_summary").toPandas()
daily = daily.sort_values("measurement_date").reset_index(drop=True)
daily["measurement_date"] = pd.to_datetime(daily["measurement_date"])

# Target: will 7-day rolling average exceed 0.45 in 1, 2, or 3 days from today?
daily["target_breach_in_3d"] = (
    daily["lead_rolling_7day_avg"]
    .shift(-3)          # look 3 days forward
    .gt(0.45)
    .astype(int)
)

# Lag features: recent lead, PM10, SO2 (1, 3, 7 days back)
for lag in [1, 3, 7]:
    daily[f"lead_lag_{lag}d"]    = daily["lead_daily_obs_ug_m3"].shift(lag)
    daily[f"pm10_lag_{lag}d"]    = daily["pm10_avg_ug_m3"].shift(lag)
    daily[f"so2_lag_{lag}d"]     = daily["so2_avg_ppm"].shift(lag)
    daily[f"wind_speed_lag_{lag}d"] = daily["wind_speed_avg"].shift(lag)

# Rolling trends (momentum features)
daily["lead_3d_trend"]  = daily["lead_daily_obs_ug_m3"].rolling(3).mean()
daily["lead_7d_trend"]  = daily["lead_daily_obs_ug_m3"].rolling(7).mean()
daily["pm10_7d_trend"]  = daily["pm10_avg_ug_m3"].rolling(7).mean()

# Wind direction: decompose into sin/cos to handle circular nature
daily["wind_dir_sin"] = np.sin(np.radians(daily["wind_direction_avg"].fillna(0)))
daily["wind_dir_cos"] = np.cos(np.radians(daily["wind_direction_avg"].fillna(0)))

# Calendar features
daily["month"]      = daily["measurement_date"].dt.month
daily["day_of_year"]= daily["measurement_date"].dt.dayofyear

FEATURES = [
    # Recent observations
    "lead_lag_1d", "lead_lag_3d", "lead_lag_7d",
    "pm10_lag_1d", "pm10_lag_3d", "pm10_lag_7d",
    "so2_lag_1d",  "so2_lag_3d",
    # Trend features
    "lead_3d_trend", "lead_7d_trend", "pm10_7d_trend",
    # Meteorology
    "wind_speed_avg", "wind_dir_sin", "wind_dir_cos",
    "temperature_avg_c", "wind_speed_lag_1d", "wind_speed_lag_3d",
    # Calendar
    "month", "day_of_year",
]

TARGET = "target_breach_in_3d"

# Drop rows with NaN in target or too many lag-feature nulls
df_model = daily.dropna(subset=[TARGET] + ["lead_lag_7d"])
print(f"Training dataset: {len(df_model)} days")
print(f"Breach rate: {df_model[TARGET].mean():.1%}")

# COMMAND ----------

# MAGIC %md ## 2. Time-Series Train / Validation Split
# MAGIC
# MAGIC Use time-aware split — never shuffle temporal data.
# MAGIC Train on earlier years, validate on most recent 12 months.

# COMMAND ----------

split_date = df_model["measurement_date"].max() - pd.DateOffset(months=12)

train = df_model[df_model["measurement_date"] <= split_date]
val   = df_model[df_model["measurement_date"] >  split_date]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]

print(f"Train: {len(train)} rows ({train['measurement_date'].min().date()} → {train['measurement_date'].max().date()})")
print(f"Val:   {len(val)}   rows ({val['measurement_date'].min().date()} → {val['measurement_date'].max().date()})")

# COMMAND ----------

# MAGIC %md ## 3. Train & Log with MLflow

# COMMAND ----------

with mlflow.start_run(run_name="lead_breach_gbt_v1") as run:
    # Pipeline: impute missing sensors → scale → gradient boosted trees
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    auc  = roc_auc_score(y_val, val_probs)
    report = classification_report(y_val, val_preds, output_dict=True)

    # Log params & metrics
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("features", FEATURES)
    mlflow.log_param("prediction_horizon_days", 3)
    mlflow.log_param("threshold_ug_m3", 0.45)

    mlflow.log_metric("val_auc_roc", round(auc, 4))
    mlflow.log_metric("val_precision_breach", round(report["1"]["precision"], 3))
    mlflow.log_metric("val_recall_breach",    round(report["1"]["recall"], 3))
    mlflow.log_metric("val_f1_breach",        round(report["1"]["f1-score"], 3))

    # Feature importance chart
    fi = pd.Series(
        pipeline.named_steps["model"].feature_importances_,
        index=FEATURES
    ).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="#E8421E")
    ax.set_title("Top Feature Importances — Lead Breach Predictor")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.show()

    # Log model to Unity Catalog
    signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name=f"{CATALOG}.gold.{MODEL_NAME}",
    )

    run_id = run.info.run_id
    print(f"\n✓ MLflow run: {run_id}")
    print(f"  Validation AUC-ROC: {auc:.4f}")
    print(f"  Precision (breach): {report['1']['precision']:.3f}")
    print(f"  Recall (breach):    {report['1']['recall']:.3f}")

# COMMAND ----------

# MAGIC %md ## 4. Apply Model → Predictions Table
# MAGIC
# MAGIC Score all historical + recent dates and write to `gold.breach_predictions`.
# MAGIC This table is refreshed daily and surfaced in the Databricks SQL dashboard.

# COMMAND ----------

# Load registered model from Unity Catalog
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Score full dataset
df_score = df_model.copy()
df_score["breach_probability_3d"] = loaded_model.predict_proba(df_score[FEATURES])[:, 1]
df_score["breach_predicted_3d"]   = (df_score["breach_probability_3d"] >= 0.5).astype(int)
df_score["risk_band"] = pd.cut(
    df_score["breach_probability_3d"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
).astype(str)

predictions_df = spark.createDataFrame(
    df_score[["measurement_date", "station_id",
              "lead_rolling_7day_avg", "target_breach_in_3d",
              "breach_probability_3d", "breach_predicted_3d", "risk_band"]]
    .rename(columns={"target_breach_in_3d": "actual_breach"})
)

(predictions_df.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.gold.breach_predictions"))

print(f"\u2713 gold.breach_predictions: {predictions_df.count():,} rows")

# Show recent forecast
display(
    spark.table(f"{CATALOG}.gold.breach_predictions")
    .orderBy("measurement_date", ascending=False)
    .limit(14)
)

# COMMAND ----------

# MAGIC %md ## 5. What drives high-risk days?
# MAGIC
# MAGIC Using Databricks AI Functions to generate a natural language explanation
# MAGIC of the most recent high-risk day — ready for the Genie demo.

# COMMAND ----------

# Pull most recent HIGH risk record for narrative
high_risk = (spark.table(f"{CATALOG}.gold.breach_predictions")
    .filter("risk_band = 'HIGH'")
    .orderBy("measurement_date", ascending=False)
    .limit(1)
    .toPandas())

if not high_risk.empty:
    row = high_risk.iloc[0]
    prompt = f"""
    Port Pirie Oliver Street monitoring station (PTP01) recorded a HIGH risk of Lead in Air
    exceeding 0.45 ug/m3 on {row['measurement_date'].date()}.
    The 7-day rolling average was {row['lead_rolling_7day_avg']:.3f} ug/m3
    with a predicted breach probability of {row['breach_probability_3d']:.0%}.

    In 2-3 sentences, explain what this means for outdoor work permits and
    what safety actions the site manager should take.
    """

    # Using Databricks AI Functions (ai_query) — runs in SQL, shown here for demo context
    explanation = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            '{prompt.replace(chr(10), " ").replace("'", "''")}'
        ) AS safety_recommendation
    """)
    display(explanation)