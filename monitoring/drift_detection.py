"""Data drift detection using Evidently AI."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset

REFERENCE_DATA_PATH = Path(__file__).parent / "reference_data.csv"


def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used during training."""
    return pd.read_csv(REFERENCE_DATA_PATH)


def _shared_non_empty_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    """Columns present in both frames with at least one non-null value on each side."""
    return [
        col
        for col in reference.columns
        if col in current.columns
        and reference[col].notna().any()
        and current[col].notna().any()
    ]


def generate_drift_report(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
) -> Snapshot:
    """Generate a data drift snapshot comparing reference and production data.

    Columns that are entirely null on either side are dropped before running the
    report — Evidently rejects empty columns.
    """
    columns = _shared_non_empty_columns(reference_data, production_data)
    report = Report(metrics=[DataDriftPreset()])
    return report.run(
        reference_data=reference_data[columns],
        current_data=production_data[columns],
    )


def save_drift_report_html(snapshot: Snapshot, output_path: str = "drift_report.html") -> None:
    """Save the drift report as an interactive HTML file."""
    snapshot.save_html(output_path)
