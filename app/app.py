import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from databricks import sql
from databricks.sdk import WorkspaceClient
from datetime import date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Port Pirie Air Quality | Compliance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATALOG = "epa_air_quality"

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "COMPLIANT":           "#2ecc71",
    "APPROACHING":         "#f39c12",
    "SITE_TARGET_BREACH":  "#e74c3c",
    "EPA_BREACH":          "#8e44ad",
    "lead":    "#E8421E",
    "pm10":    "#3498db",
    "so2":     "#9b59b6",
    "target":  "#f39c12",
    "limit":   "#e74c3c",
    "LOW":     "#2ecc71",
    "MEDIUM":  "#f39c12",
    "HIGH":    "#e74c3c",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        border-radius: 10px; padding: 20px; text-align: center;
        border: 1px solid #e0e0e0; margin-bottom: 10px;
    }
    .metric-label  { font-size: 13px; color: #666; text-transform: uppercase;
                     letter-spacing: 1px; margin-bottom: 4px; }
    .metric-value  { font-size: 32px; font-weight: 700; line-height: 1.1; }
    .metric-sub    { font-size: 12px; color: #888; margin-top: 6px; }
    .status-badge  { display:inline-block; padding: 4px 12px; border-radius: 20px;
                     font-size: 12px; font-weight: 600; color: white; }
    .sf-badge      { background: #00a1e0; color: white; border-radius: 4px;
                     padding: 2px 8px; font-size: 11px; font-weight: 600; }
    .section-header { font-size: 15px; font-weight: 600; color: #333;
                      border-left: 4px solid #E8421E; padding-left: 10px;
                      margin: 20px 0 12px 0; }
    div[data-testid="stTabs"] button { font-size: 14px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Databricks connection ─────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    """Auto-authenticates via Databricks App runtime environment."""
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_WAREHOUSE_HTTP_PATH"],
        credentials_provider=lambda: {"token": [os.environ["DATABRICKS_TOKEN"]]},
    )

@st.cache_data(ttl=300)
def query(sql_str: str) -> pd.DataFrame:
    with get_connection().cursor() as cur:
        cur.execute(sql_str)
        return cur.fetchall_arrow().to_pandas()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/6/63/Databricks_Logo.png",
        width=160,
    )
    st.markdown("---")
    st.markdown("**Station**")
    station = st.selectbox("Monitoring station", ["PTP01 — Oliver Street"], index=0)
    station_id = "PTP01"

    st.markdown("**Date range**")
    date_to   = st.date_input("To",   value=date.today())
    date_from = st.date_input("From", value=date_to - timedelta(days=365))

    st.markdown("---")
    st.markdown(
        '<span class="sf-badge">SF</span> License data via Salesforce Data360',
        unsafe_allow_html=True,
    )
    st.caption("EPA limits are read directly from the Nyrstar Port Pirie business license managed in Salesforce.")
    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("## 🏭 Port Pirie Air Quality — Compliance Dashboard")
    st.caption(
        f"**Nyrstar Port Pirie** · Oliver Street Station (PTP01) · "
        f"{date_from.strftime('%d %b %Y')} → {date_to.strftime('%d %b %Y')}"
    )

st.markdown("---")

# ── Load core data ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_permit_status():
    return query(f"""
        SELECT pollutant, current_value, unit, epa_limit, site_target,
               pct_of_site_target, permit_status, action_required,
               averaging_period, assessed_at
        FROM {CATALOG}.gold.permit_status_current
        WHERE station_id = 'PTP01'
    """)

@st.cache_data(ttl=300)
def load_lead_rolling(date_from, date_to):
    return query(f"""
        SELECT measurement_date, lead_in_air_ug_m3, rolling_7day_avg_lead_ug_m3,
               site_target, limit_value AS epa_limit,
               compliance_status, trigger_investigation, pct_of_site_target
        FROM {CATALOG}.gold.lead_rolling_weekly_avg
        WHERE station_id = 'PTP01'
          AND measurement_date BETWEEN '{date_from}' AND '{date_to}'
        ORDER BY measurement_date
    """)

@st.cache_data(ttl=300)
def load_daily(date_from, date_to):
    return query(f"""
        SELECT measurement_date, pm10_avg_ug_m3, pm10_max_ug_m3,
               so2_avg_ppm, so2_max_ppm,
               wind_speed_avg, wind_direction_avg, temperature_avg_c,
               lead_daily_obs_ug_m3, lead_rolling_7day_avg
        FROM {CATALOG}.gold.pollutant_daily_summary
        WHERE station_id = 'PTP01'
          AND measurement_date BETWEEN '{date_from}' AND '{date_to}'
        ORDER BY measurement_date
    """)

@st.cache_data(ttl=300)
def load_predictions(date_from, date_to):
    return query(f"""
        SELECT measurement_date, lead_rolling_7day_avg,
               breach_probability_3d, risk_band, actual_breach
        FROM {CATALOG}.gold.breach_predictions
        WHERE measurement_date BETWEEN '{date_from}' AND '{date_to}'
        ORDER BY measurement_date
    """)

@st.cache_data(ttl=300)
def load_exceedances(date_from, date_to):
    return query(f"""
        SELECT event_timestamp, pollutant, observed_value, unit,
               threshold, pct_of_threshold
        FROM {CATALOG}.gold.exceedance_log
        WHERE event_timestamp BETWEEN '{date_from}' AND '{date_to}'
        ORDER BY event_timestamp DESC
        LIMIT 200
    """)

@st.cache_data(ttl=300)
def load_monthly():
    return query(f"""
        SELECT month, lead_monthly_avg, lead_p95, lead_max,
               days_site_target_breached, pm10_monthly_avg, so2_monthly_avg
        FROM {CATALOG}.gold.pollutant_monthly_stats
        WHERE station_id = 'PTP01'
        ORDER BY month
    """)

@st.cache_data(ttl=300)
def load_advisories():
    return query(f"""
        SELECT advisory_date, risk_band, rolling_7day_avg,
               breach_probability, advisory_text, generated_at
        FROM {CATALOG}.gold.school_air_quality_advisories
        ORDER BY advisory_date DESC
    """)

@st.cache_data(ttl=300)
def load_sf_license():
    return query(f"""
        SELECT pollutant, limit_value, site_target, unit, averaging_period
        FROM {CATALOG}.salesforce.epa_limits
        WHERE station_id = 'PTP01'
    """)

with st.spinner("Loading data…"):
    df_status      = load_permit_status()
    df_lead        = load_lead_rolling(date_from, date_to)
    df_daily       = load_daily(date_from, date_to)
    df_predictions = load_predictions(date_from, date_to)
    df_exceedances = load_exceedances(date_from, date_to)
    df_monthly     = load_monthly()
    df_license     = load_sf_license()
    df_advisories  = load_advisories()

# convert dates
for df, col in [(df_lead, "measurement_date"), (df_daily, "measurement_date"),
                (df_predictions, "measurement_date"), (df_monthly, "month")]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_lead, tab_pollutants, tab_ml, tab_log, tab_ai, tab_schools = st.tabs([
    "📊 Overview",
    "🔬 Lead in Air",
    "💨 PM10 & SO2",
    "🤖 ML Prediction",
    "📋 Exceedance Log",
    "✨ AI Assistant",
    "🏫 Schools Advisory",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Permit status cards ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Current Permit Status</div>', unsafe_allow_html=True)
    st.caption("Limits sourced from Salesforce business license · Station PTP01")

    cols = st.columns(3)
    for i, row in df_status.iterrows():
        colour = COLOURS.get(row["permit_status"], "#95a5a6")
        icon   = "✅" if row["permit_status"] == "COMPLIANT" else ("⚠️" if "APPROACHING" in row["permit_status"] else "🚨")
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {colour};">
                <div class="metric-label">{row['pollutant']}</div>
                <div class="metric-value" style="color:{colour};">
                    {row['current_value']:.3g} <span style="font-size:16px">{row['unit']}</span>
                </div>
                <div class="metric-sub">
                    {icon} <span class="status-badge" style="background:{colour};">{row['permit_status'].replace('_',' ')}</span>
                </div>
                <div class="metric-sub" style="margin-top:10px">
                    Site target: <b>{row['site_target']:.3g}</b> &nbsp;|&nbsp;
                    EPA limit: <b>{row['epa_limit']:.3g}</b> {row['unit']}<br>
                    <b>{row['pct_of_site_target']:.0f}%</b> of site target &nbsp;·&nbsp;
                    {row['averaging_period'].replace('_',' ')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Salesforce license context ────────────────────────────────────────────
    with st.expander("🏢 Salesforce Business License — Nyrstar Port Pirie", expanded=False):
        st.caption("This data is shared from Salesforce to Databricks via Data360")
        st.dataframe(
            df_license.rename(columns={
                "pollutant": "Pollutant", "limit_value": "EPA Limit",
                "site_target": "Site Target", "unit": "Unit",
                "averaging_period": "Averaging Period"
            }),
            use_container_width=True, hide_index=True,
        )

    # ── Lead rolling avg — summary chart ─────────────────────────────────────
    st.markdown('<div class="section-header">Lead in Air — 7-Day Rolling Average</div>', unsafe_allow_html=True)

    if not df_lead.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_lead["measurement_date"], y=df_lead["lead_in_air_ug_m3"],
            name="Daily obs", marker_color="#adb5bd", opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=df_lead["measurement_date"], y=df_lead["rolling_7day_avg_lead_ug_m3"],
            name="7-day rolling avg", line=dict(color=COLOURS["lead"], width=2.5),
        ))
        fig.add_hline(y=0.45, line_dash="dot", line_color=COLOURS["target"],
                      annotation_text="Site target 0.45 μg/m³", annotation_position="top left")
        fig.add_hline(y=0.50, line_dash="dash", line_color=COLOURS["limit"],
                      annotation_text="EPA limit 0.50 μg/m³", annotation_position="top left")
        # Shade breach zones
        breach = df_lead[df_lead["trigger_investigation"]]
        if not breach.empty:
            for _, br in breach.iterrows():
                fig.add_vrect(
                    x0=br["measurement_date"] - timedelta(hours=12),
                    x1=br["measurement_date"] + timedelta(hours=12),
                    fillcolor="rgba(231,76,60,0.1)", line_width=0,
                )
        fig.update_layout(
            height=320, margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Lead in Air (μg/m³)", xaxis_title=None,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    breach_days = int(df_lead["trigger_investigation"].sum()) if not df_lead.empty else 0
    max_rolling = df_lead["rolling_7day_avg_lead_ug_m3"].max() if not df_lead.empty else 0
    exc_count   = len(df_exceedances)
    k1.metric("Days site target breached", breach_days)
    k2.metric("Max 7-day rolling avg", f"{max_rolling:.3f} μg/m³")
    k3.metric("Total exceedance events", exc_count)
    k4.metric("Date range", f"{(date_to - date_from).days} days")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LEAD IN AIR (DETAILED)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_lead:

    # ── Rolling average detail ────────────────────────────────────────────────
    st.markdown('<div class="section-header">7-Day Rolling Average vs EPA Limits</div>', unsafe_allow_html=True)
    st.caption("Salesforce Data360 activation triggers an investigation case when rolling avg > 0.45 μg/m³")

    if not df_lead.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.06)

        # Top: rolling avg
        fig.add_trace(go.Scatter(
            x=df_lead["measurement_date"], y=df_lead["lead_in_air_ug_m3"],
            name="Daily obs", mode="markers",
            marker=dict(color="#adb5bd", size=4, opacity=0.6),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_lead["measurement_date"], y=df_lead["rolling_7day_avg_lead_ug_m3"],
            name="7-day rolling avg", line=dict(color=COLOURS["lead"], width=2.5),
            fill="tozeroy", fillcolor="rgba(232,66,30,0.08)",
        ), row=1, col=1)
        fig.add_hline(y=0.45, line_dash="dot", line_color=COLOURS["target"], row=1, col=1,
                      annotation_text="Site target 0.45", annotation_position="top left")
        fig.add_hline(y=0.50, line_dash="dash", line_color=COLOURS["limit"], row=1, col=1,
                      annotation_text="EPA limit 0.50", annotation_position="top left")

        # Bottom: % of site target
        colours_pct = df_lead["pct_of_site_target"].apply(
            lambda v: COLOURS["limit"] if v > 100 else (COLOURS["target"] if v > 85 else COLOURS["COMPLIANT"])
        )
        fig.add_trace(go.Bar(
            x=df_lead["measurement_date"], y=df_lead["pct_of_site_target"],
            name="% of site target", marker_color=colours_pct,
        ), row=2, col=1)
        fig.add_hline(y=100, line_dash="dot", line_color=COLOURS["target"], row=2, col=1)

        fig.update_yaxes(title_text="Lead in Air (μg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="% of site target", row=2, col=1)
        fig.update_layout(height=520, margin=dict(t=20, b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # ── Monthly trend ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Monthly Performance</div>', unsafe_allow_html=True)

    if not df_monthly.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_monthly["month"], y=df_monthly["lead_monthly_avg"],
            name="Monthly avg", marker_color=COLOURS["lead"], opacity=0.75,
        ))
        fig2.add_trace(go.Scatter(
            x=df_monthly["month"], y=df_monthly["lead_p95"],
            name="P95", line=dict(color=COLOURS["target"], dash="dot"), mode="lines",
        ))
        fig2.add_hline(y=0.45, line_dash="dot", line_color=COLOURS["target"],
                       annotation_text="Site target 0.45")
        fig2.add_hline(y=0.50, line_dash="dash", line_color=COLOURS["limit"],
                       annotation_text="EPA limit 0.50")
        fig2.update_layout(height=300, margin=dict(t=20, b=20),
                           yaxis_title="Lead in Air (μg/m³)", hovermode="x unified",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

        # Breach day count bar
        fig3 = go.Figure(go.Bar(
            x=df_monthly["month"], y=df_monthly["days_site_target_breached"],
            marker_color=COLOURS["SITE_TARGET_BREACH"], name="Days target breached",
        ))
        fig3.update_layout(height=180, margin=dict(t=10, b=20),
                           yaxis_title="Days breached", title="Days site target breached per month")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PM10 & SO2
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pollutants:

    if not df_daily.empty:
        # ── PM10 ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">PM10 — Daily Average & Peak vs EPA Limit (50 μg/m³)</div>',
                    unsafe_allow_html=True)

        fig_pm10 = go.Figure()
        fig_pm10.add_trace(go.Scatter(
            x=df_daily["measurement_date"], y=df_daily["pm10_max_ug_m3"],
            name="Daily max", line=dict(color="#adb5bd", width=1),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.07)",
        ))
        fig_pm10.add_trace(go.Scatter(
            x=df_daily["measurement_date"], y=df_daily["pm10_avg_ug_m3"],
            name="Daily avg", line=dict(color=COLOURS["pm10"], width=2),
        ))
        fig_pm10.add_hline(y=50, line_dash="dash", line_color=COLOURS["limit"],
                           annotation_text="EPA limit 50 μg/m³")
        fig_pm10.update_layout(height=300, margin=dict(t=20, b=20),
                               yaxis_title="PM10 (μg/m³)", hovermode="x unified",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_pm10, use_container_width=True)

        # ── SO2 ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">SO2 — Daily Average & Peak vs EPA Limit (0.20 ppm)</div>',
                    unsafe_allow_html=True)

        fig_so2 = go.Figure()
        fig_so2.add_trace(go.Scatter(
            x=df_daily["measurement_date"], y=df_daily["so2_max_ppm"],
            name="Daily max", line=dict(color="#adb5bd", width=1),
            fill="tozeroy", fillcolor="rgba(155,89,182,0.07)",
        ))
        fig_so2.add_trace(go.Scatter(
            x=df_daily["measurement_date"], y=df_daily["so2_avg_ppm"],
            name="Daily avg", line=dict(color=COLOURS["so2"], width=2),
        ))
        fig_so2.add_hline(y=0.20, line_dash="dash", line_color=COLOURS["limit"],
                          annotation_text="EPA limit 0.20 ppm")
        fig_so2.update_layout(height=300, margin=dict(t=20, b=20),
                              yaxis_title="SO2 (ppm)", hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_so2, use_container_width=True)

        # ── Wind / met context ───────────────────────────────────────────────
        st.markdown('<div class="section-header">Meteorology Context</div>', unsafe_allow_html=True)
        col_ws, col_tmp = st.columns(2)

        with col_ws:
            fig_wind = go.Figure(go.Scatter(
                x=df_daily["measurement_date"], y=df_daily["wind_speed_avg"],
                fill="tozeroy", line=dict(color="#17a2b8", width=1.5),
                name="Wind speed avg",
            ))
            fig_wind.update_layout(height=220, margin=dict(t=10, b=10),
                                   yaxis_title="Wind speed (m/s)", title="Wind Speed")
            st.plotly_chart(fig_wind, use_container_width=True)

        with col_tmp:
            fig_temp = go.Figure(go.Scatter(
                x=df_daily["measurement_date"], y=df_daily["temperature_avg_c"],
                fill="tozeroy", line=dict(color="#fd7e14", width=1.5),
                name="Temperature",
            ))
            fig_temp.update_layout(height=220, margin=dict(t=10, b=10),
                                   yaxis_title="°C", title="Temperature")
            st.plotly_chart(fig_temp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ml:

    st.markdown('<div class="section-header">ML Model — Lead Breach Probability (3-Day Horizon)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Gradient Boosted Tree model trained on 3 years of historical data. "
        "Predicts probability that the 7-day rolling Lead in Air average will exceed "
        "0.45 μg/m³ within the next 3 days. Features: wind speed/direction, temperature, "
        "humidity, lagged PM10/SO2/Lead readings."
    )

    if not df_predictions.empty:
        # ── Latest risk callout ───────────────────────────────────────────────
        latest = df_predictions.iloc[-1]
        risk   = latest["risk_band"]
        prob   = latest["breach_probability_3d"] * 100
        colour = COLOURS.get(risk, "#95a5a6")

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {colour};">
            <div class="metric-label">Current ML Risk Band</div>
            <div class="metric-value" style="color:{colour};">{risk}</div>
            <div class="metric-sub">as at {latest['measurement_date'].date()}</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {colour};">
            <div class="metric-label">Breach Probability (3d)</div>
            <div class="metric-value" style="color:{colour};">{prob:.0f}%</div>
            <div class="metric-sub">rolling avg > 0.45 μg/m³</div>
        </div>
        """, unsafe_allow_html=True)
        with c3:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": colour},
                    "steps": [
                        {"range": [0, 30],  "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fef9e7"},
                        {"range": [60, 100],"color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": COLOURS["limit"], "width": 3},
                                  "value": 50},
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=10, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Probability time-series ───────────────────────────────────────────
        colour_seq = df_predictions["risk_band"].map(COLOURS).fillna("#95a5a6")

        fig_pred = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.65, 0.35], vertical_spacing=0.06)
        fig_pred.add_trace(go.Scatter(
            x=df_predictions["measurement_date"],
            y=df_predictions["breach_probability_3d"] * 100,
            name="Breach probability", fill="tozeroy",
            line=dict(color=COLOURS["lead"], width=2),
            fillcolor="rgba(232,66,30,0.1)",
        ), row=1, col=1)
        fig_pred.add_hline(y=50, line_dash="dash", line_color=COLOURS["limit"],
                           annotation_text="Decision threshold 50%", row=1, col=1)
        # Actual vs predicted
        fig_pred.add_trace(go.Scatter(
            x=df_predictions["measurement_date"],
            y=df_predictions["actual_breach"].astype(int),
            name="Actual breach", mode="markers",
            marker=dict(color=COLOURS["limit"], size=5, symbol="x"),
        ), row=2, col=1)
        fig_pred.add_trace(go.Scatter(
            x=df_predictions["measurement_date"],
            y=df_predictions["breach_probability_3d"],
            name="Predicted prob", line=dict(color=COLOURS["lead"], width=1.5),
        ), row=2, col=1)

        fig_pred.update_yaxes(title_text="Breach probability (%)", row=1, col=1)
        fig_pred.update_yaxes(title_text="Actual vs Pred", row=2, col=1)
        fig_pred.update_layout(height=480, margin=dict(t=20, b=20), hovermode="x unified",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── Model info ────────────────────────────────────────────────────────
        with st.expander("ℹ️ Model details"):
            st.markdown("""
            | | |
            |---|---|
            | **Algorithm** | Gradient Boosted Tree (sklearn) |
            | **Prediction horizon** | 3 days |
            | **Target** | 7-day rolling Lead in Air avg > 0.45 μg/m³ |
            | **Key features** | Wind speed/direction, temperature, lagged Lead/PM10/SO2 readings |
            | **Tracked in** | MLflow · Unity Catalog Model Registry |
            | **Registered model** | `epa_air_quality.gold.port_pirie_lead_breach_predictor` |
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXCEEDANCE LOG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_log:

    st.markdown('<div class="section-header">Threshold Exceedance Events</div>', unsafe_allow_html=True)
    st.caption(f"{len(df_exceedances)} events in selected period · Immutable audit log in Delta Lake")

    if not df_exceedances.empty:
        pollutant_filter = st.multiselect(
            "Filter by pollutant",
            options=df_exceedances["pollutant"].unique().tolist(),
            default=df_exceedances["pollutant"].unique().tolist(),
        )
        filtered = df_exceedances[df_exceedances["pollutant"].isin(pollutant_filter)]

        # Colour severity
        def severity(pct):
            if pct > 150: return "🔴 Critical"
            if pct > 110: return "🟠 Breach"
            return "🟡 Approaching"

        filtered = filtered.copy()
        filtered["severity"] = filtered["pct_of_threshold"].apply(severity)
        filtered["observed_value"] = filtered["observed_value"].round(4)
        filtered["pct_of_threshold"] = filtered["pct_of_threshold"].round(1)

        st.dataframe(
            filtered.rename(columns={
                "event_timestamp": "Date/Time", "pollutant": "Pollutant",
                "observed_value": "Observed", "unit": "Unit",
                "threshold": "Site Target", "pct_of_threshold": "% of Target",
                "severity": "Severity",
            }),
            use_container_width=True, hide_index=True,
        )

        # Exceedances by pollutant bar
        counts = filtered.groupby("pollutant").size().reset_index(name="count")
        fig_exc = px.bar(counts, x="pollutant", y="count", color="pollutant",
                         color_discrete_map={"Lead in Air": COLOURS["lead"],
                                             "PM10": COLOURS["pm10"],
                                             "SO2": COLOURS["so2"]},
                         title="Exceedance events by pollutant")
        fig_exc.update_layout(height=250, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_exc, use_container_width=True)
    else:
        st.success("✅ No threshold exceedances in the selected period.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ai:

    st.markdown('<div class="section-header">AI Assistant — Powered by Databricks AI Functions</div>',
                unsafe_allow_html=True)
    st.caption(
        "Ask questions in plain English about the Port Pirie air quality data. "
        "Responses are generated using `ai_query()` against Databricks Foundation Models, "
        "grounded in the actual monitoring data."
    )

    # ── Pre-built questions ───────────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    quick_qs = [
        "Summarise the current compliance status for Nyrstar Port Pirie.",
        "When was the most recent Lead in Air breach and how long did it last?",
        "What safety actions should the site manager take today given the current risk band?",
        "How does PM10 performance compare across the last 12 months?",
        "What meteorological conditions are associated with the highest lead readings?",
    ]
    selected_q = st.selectbox("Choose a question or type your own below:", [""] + quick_qs)

    user_q = st.text_area(
        "Your question:",
        value=selected_q,
        placeholder="e.g. Were there any SO2 exceedances in summer 2023?",
        height=80,
    )

    if st.button("Ask AI", type="primary") and user_q:
        with st.spinner("Querying Databricks AI…"):
            # Build context from live data
            status_summary = df_status[["pollutant","current_value","unit","permit_status"]].to_string(index=False)
            lead_latest = df_lead.tail(14).to_string(index=False) if not df_lead.empty else "No data"
            breach_count = int(df_lead["trigger_investigation"].sum()) if not df_lead.empty else 0
            latest_risk = df_predictions.iloc[-1]["risk_band"] if not df_predictions.empty else "Unknown"
            latest_prob = df_predictions.iloc[-1]["breach_probability_3d"] if not df_predictions.empty else 0

            context = f"""
You are an environmental compliance advisor for Port Pirie, South Australia.
Nyrstar Port Pirie operates a multi-metal smelter and refinery under a business license.
Monitoring station PTP01 (Oliver Street) tracks Lead in Air, PM10 and SO2.

EPA limits (from Salesforce business license):
- Lead in Air: site target 0.45 μg/m³, EPA limit 0.50 μg/m³ (7-day rolling average)
- PM10: 50 μg/m³ (24-hour average)
- SO2: 0.20 ppm (1-hour average)

Current permit status:
{status_summary}

Last 14 days of Lead in Air (rolling avg):
{lead_latest}

Site target breach days in selected period: {breach_count}
ML model current risk band: {latest_risk} ({latest_prob*100:.0f}% breach probability in next 3 days)
Total exceedance events in period: {len(df_exceedances)}

Answer the following question concisely and in plain English:
{user_q}
"""
            result = query(f"""
                SELECT ai_query(
                    'databricks-meta-llama-3-1-70b-instruct',
                    '{context.replace(chr(39), chr(39)+chr(39))}'
                ) AS response
            """)
            if not result.empty:
                st.markdown(f"""
                <div style="background:#f8f9fa; border-left:4px solid #E8421E;
                            padding:16px; border-radius:4px; margin-top:12px;">
                    {result.iloc[0]['response']}
                </div>
                """, unsafe_allow_html=True)

    # ── Genie embed note ──────────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "**Tip:** For full natural-language data exploration, embed a **Databricks AI/BI Genie** space "
        "pointed at the `epa_air_quality.gold` schema. Genie allows analysts to ask free-form "
        "questions like *'Show me all days where lead exceeded the target while wind was from the north-west'* "
        "and get back charts automatically."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SCHOOLS ADVISORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_schools:

    st.markdown('<div class="section-header">School Community Air Quality Advisories</div>',
                unsafe_allow_html=True)

    # ── How it works ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ How advisories are generated", expanded=False):
        st.markdown("""
        1. **ML model** (notebook 04) scores each day's breach probability using wind, temperature and lagged air quality readings
        2. Days classified as **HIGH** or **MEDIUM** risk trigger advisory generation
        3. **Databricks `ai_query()`** calls Meta LLaMA 3.3 70B with the risk level, rolling average, and breach probability as context
        4. The model writes a plain-English advisory addressed to school principals with specific outdoor activity guidance
        5. Advisories are persisted to `epa_air_quality.gold.school_air_quality_advisories` via MERGE (re-runs update, not duplicate)
        6. **Compliant days produce no advisory** — the absence of a record means conditions are safe
        """)

    if df_advisories.empty:
        st.success(
            "✅ No advisories on record — Lead in Air levels are within safe limits. "
            "Run notebook 04 to generate advisories for HIGH and MEDIUM risk days."
        )
    else:
        df_advisories["advisory_date"]     = pd.to_datetime(df_advisories["advisory_date"])
        df_advisories["breach_probability"] = (df_advisories["breach_probability"] * 100).round(0)

        # ── Current status banner ─────────────────────────────────────────────
        # Check if there is a recent advisory (within 7 days of most recent lead data)
        most_recent_lead_date = df_lead["measurement_date"].max() if not df_lead.empty else pd.Timestamp.now()
        recent_cutoff         = most_recent_lead_date - pd.Timedelta(days=7)
        active_advisory       = df_advisories[df_advisories["advisory_date"] >= recent_cutoff]

        if active_advisory.empty:
            st.success(
                f"✅ **No active advisory** — Lead in Air levels have been within safe limits "
                f"for the past 7 days. Schools may operate normal outdoor schedules."
            )
        else:
            latest = active_advisory.iloc[0]
            risk   = latest["risk_band"]
            colour = COLOURS.get(risk, "#95a5a6")
            icon   = "🔴" if risk == "HIGH" else "🟠"
            banner_bg = "#fff5f5" if risk == "HIGH" else "#fffbf0"
            st.markdown(f"""
            <div style="background:{banner_bg}; border:2px solid {colour};
                        border-radius:8px; padding:16px; margin-bottom:16px;">
                <span style="font-size:20px">{icon}</span>
                <b style="font-size:16px; color:{colour}"> ACTIVE {risk} ADVISORY</b>
                &nbsp;·&nbsp; {latest['advisory_date'].strftime('%A %d %B %Y')}
                &nbsp;·&nbsp; 7-day avg: <b>{latest['rolling_7day_avg']:.3f} μg/m³</b>
                &nbsp;·&nbsp; Breach probability: <b>{latest['breach_probability']:.0f}%</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Full timeline: rolling avg + advisory markers ─────────────────────
        st.markdown('<div class="section-header">Lead in Air Timeline — Advisory Events in Context</div>',
                    unsafe_allow_html=True)
        st.caption("Line = full Lead in Air 7-day rolling average across all dates · Markers = days an advisory was issued · Gaps = compliant, no advisory")

        fig_timeline = go.Figure()

        # Background: full rolling avg line (all dates, not just advisory days)
        if not df_lead.empty:
            fig_timeline.add_trace(go.Scatter(
                x=df_lead["measurement_date"],
                y=df_lead["rolling_7day_avg_lead_ug_m3"],
                name="7-day rolling avg (all days)",
                line=dict(color="#adb5bd", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(173,181,189,0.1)",
                hovertemplate="%{x|%d %b %Y}<br>Rolling avg: %{y:.3f} μg/m³<extra></extra>",
            ))

        # Advisory event markers overlaid on the line
        for band, marker_sym, size in [("HIGH", "circle", 14), ("MEDIUM", "diamond", 12)]:
            sub = df_advisories[df_advisories["risk_band"] == band]
            if not sub.empty:
                fig_timeline.add_trace(go.Scatter(
                    x=sub["advisory_date"],
                    y=sub["rolling_7day_avg"],
                    mode="markers",
                    name=f"{band} advisory issued",
                    marker=dict(color=COLOURS[band], size=size, symbol=marker_sym,
                                line=dict(color="white", width=2)),
                    customdata=sub[["breach_probability", "risk_band"]],
                    hovertemplate=(
                        "<b>Advisory issued: %{x|%d %b %Y}</b><br>"
                        "Risk: %{customdata[1]}<br>"
                        "7-day avg: %{y:.3f} μg/m³<br>"
                        "Breach prob: %{customdata[0]:.0f}%<br>"
                        "<extra></extra>"
                    ),
                ))

        fig_timeline.add_hline(y=0.45, line_dash="dot", line_color=COLOURS["target"],
                               annotation_text="Site target 0.45 μg/m³", annotation_position="top left")
        fig_timeline.add_hline(y=0.50, line_dash="dash", line_color=COLOURS["limit"],
                               annotation_text="EPA limit 0.50 μg/m³", annotation_position="top left")
        fig_timeline.update_layout(
            height=340, margin=dict(t=20, b=20),
            yaxis_title="Lead in Air (μg/m³)", xaxis_title=None,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # ── Advisory reader ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">Read an Advisory</div>', unsafe_allow_html=True)

        c_select, c_meta = st.columns([2, 1])
        with c_select:
            selected_date = st.selectbox(
                "Select advisory date:",
                options=df_advisories["advisory_date"].dt.strftime("%d %b %Y").tolist(),
                label_visibility="collapsed",
            )

        selected_row = df_advisories[
            df_advisories["advisory_date"].dt.strftime("%d %b %Y") == selected_date
        ].iloc[0]

        sel_colour = COLOURS.get(selected_row["risk_band"], "#95a5a6")
        sel_icon   = "🔴" if selected_row["risk_band"] == "HIGH" else "🟠"

        with c_meta:
            st.markdown(
                f'{sel_icon} &nbsp;<span class="status-badge" style="background:{sel_colour};">'
                f'{selected_row["risk_band"]}</span>'
                f'&nbsp; {selected_row["rolling_7day_avg"]:.3f} μg/m³ &nbsp;·&nbsp; '
                f'{selected_row["breach_probability"]:.0f}% breach prob',
                unsafe_allow_html=True,
            )

        st.markdown(f"""
        <div style="background:#f8f9fa; border-left:4px solid {sel_colour};
                    padding:20px; border-radius:4px; font-size:15px; line-height:1.8;
                    margin-top:8px;">
            {selected_row['advisory_text']}
        </div>
        <p style="font-size:11px; color:#999; margin-top:6px;">
            ✨ Generated by Databricks <code>ai_query()</code> · meta-llama-3-3-70b-instruct ·
            {pd.to_datetime(selected_row['generated_at']).strftime('%d %b %Y %H:%M')}
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">All Advisories</div>', unsafe_allow_html=True)

        display_df = df_advisories[["advisory_date", "risk_band", "rolling_7day_avg", "breach_probability"]].copy()
        display_df["advisory_date"]     = display_df["advisory_date"].dt.strftime("%d %b %Y")
        display_df["rolling_7day_avg"]  = display_df["rolling_7day_avg"].round(3)
        display_df["breach_probability"]= display_df["breach_probability"].astype(int)
        display_df.columns              = ["Date", "Risk", "7-day avg (μg/m³)", "Breach prob (%)"]

        st.dataframe(display_df, use_container_width=True, hide_index=True)
