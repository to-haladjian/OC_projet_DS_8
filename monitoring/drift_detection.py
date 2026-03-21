"""Data drift detection using Evidently AI."""

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

REFERENCE_DATA_PATH = Path(__file__).parent / "reference_data.csv"


def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used during training."""
    return pd.read_csv(REFERENCE_DATA_PATH)


def generate_drift_report(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
) -> Report:
    """Generate a data drift report comparing reference and production data."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=production_data)
    return report


def save_drift_report_html(report: Report, output_path: str = "drift_report.html") -> None:
    """Save the drift report as an interactive HTML file."""
    report.save_html(output_path)
