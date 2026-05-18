"""Streamlit monitoring dashboard for the credit-scoring API.

Reads prediction logs from the production database (Supabase / PostgreSQL via
SQLAlchemy), surfaces KPIs and distributions, and triggers an Evidently data
drift report against the training reference set on demand.

Run:
    pip install -r requirements-monitoring.txt
    streamlit run monitoring/dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.services.db_service import get_recent_predictions  # noqa: E402
from app.services.preprocessing import build_feature_dataframe  # noqa: E402
from monitoring.drift_detection import (  # noqa: E402
    generate_drift_report,
    load_reference_data,
)

st.set_page_config(
    page_title="Credit Scoring - Monitoring",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(ttl=60)
def fetch_predictions(limit: int) -> pd.DataFrame:
    rows = get_recent_predictions(limit=limit)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def build_production_features(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = [build_feature_dataframe(dict(features)) for features in predictions["input_features"]]
    return pd.concat(frames, ignore_index=True)


st.title("Credit Scoring - Production Monitoring")
st.caption("Live KPIs and data-drift detection for the deployed model.")

with st.sidebar:
    st.header("Filters")
    limit = st.slider("Rows to fetch", min_value=50, max_value=5000, value=500, step=50)
    default_start = (datetime.utcnow() - timedelta(days=30)).date()
    date_range = st.date_input(
        "Date range",
        value=(default_start, datetime.utcnow().date()),
    )
    refresh = st.button("Refresh data")

if refresh:
    fetch_predictions.clear()

predictions = fetch_predictions(limit)

if predictions.empty:
    st.warning(
        "No prediction logs available. Make sure `DATABASE_URL` is set and the API "
        "has served at least one request."
    )
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask = (predictions["timestamp"].dt.date >= start) & (
        predictions["timestamp"].dt.date <= end
    )
    predictions = predictions.loc[mask]

if predictions.empty:
    st.info("No predictions in the selected date range.")
    st.stop()

total = len(predictions)
approval_rate = predictions["credit_approved"].mean()
latency_mean = predictions["execution_time_ms"].mean()
latency_p50 = predictions["execution_time_ms"].median()
latency_p95 = predictions["execution_time_ms"].quantile(0.95)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total predictions", f"{total:,}")
c2.metric("Approval rate", f"{approval_rate:.1%}")
c3.metric("Latency mean (ms)", f"{latency_mean:.1f}")
c4.metric("Latency p50 (ms)", f"{latency_p50:.1f}")
c5.metric("Latency p95 (ms)", f"{latency_p95:.1f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Default probability distribution")
    fig = px.histogram(
        predictions,
        x="default_probability",
        nbins=30,
        labels={"default_probability": "Predicted default probability"},
    )
    fig.update_layout(bargap=0.05, height=350)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Latency distribution")
    fig = px.histogram(
        predictions,
        x="execution_time_ms",
        nbins=30,
        labels={"execution_time_ms": "Execution time (ms)"},
    )
    fig.update_layout(bargap=0.05, height=350)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Predictions over time")
hourly = (
    predictions.set_index("timestamp")
    .resample("1h")
    .size()
    .reset_index(name="count")
)
fig = px.line(hourly, x="timestamp", y="count", markers=True)
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Data drift report")
st.caption(
    "Compare the in-window production inputs against the training reference set "
    "using Evidently. Generation may take a few seconds."
)

if st.button("Generate drift report"):
    with st.spinner("Running Evidently report..."):
        reference = load_reference_data()
        production = build_production_features(predictions)
        snapshot = generate_drift_report(reference, production)
        html = snapshot.get_html_str(as_iframe=False)
    st.components.v1.html(html, height=900, scrolling=True)

with st.expander("Raw prediction rows"):
    st.dataframe(
        predictions.drop(columns=["input_features"]).sort_values(
            "timestamp", ascending=False
        ),
        use_container_width=True,
    )
