"""Export utilities for CSV and PDF reports."""

from __future__ import annotations

import csv
import io
import os
from datetime import datetime

import pandas as pd

from evaluation.metrics import MetricsResult


def metrics_to_dataframe(results: list[MetricsResult]) -> pd.DataFrame:
    """Convert a list of MetricsResult into a DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


def export_csv(results: list[MetricsResult], output_path: str | None = None) -> str:
    """
    Export benchmark results to CSV.

    Returns the CSV as a string. If output_path is given, also writes to file.
    """
    df = metrics_to_dataframe(results)
    csv_str = df.to_csv(index=False)

    if output_path:
        df.to_csv(output_path, index=False)

    return csv_str


def export_csv_bytes(results: list[MetricsResult]) -> bytes:
    """Export benchmark results to CSV bytes (for Streamlit download)."""
    csv_str = export_csv(results)
    return csv_str.encode("utf-8")


def export_html_report(
    results: list[MetricsResult],
    title: str = "ASR Benchmark Report",
) -> str:
    """Generate an HTML report from benchmark results."""
    df = metrics_to_dataframe(results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build summary table: pivot by model
    summary_cols = ["model_name", "wer_overall", "wer_arabic", "wer_english", "rtf"]
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df[available_cols].groupby("model_name").mean(numeric_only=True).round(4)

    html = f"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; }}
h2 {{ color: #16213e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #1a1a2e; color: white; }}
tr:hover {{ background: #f0f0f0; }}
.timestamp {{ color: #666; font-size: 0.9em; }}
.metric-good {{ color: #27ae60; font-weight: bold; }}
.metric-bad {{ color: #e74c3c; font-weight: bold; }}
.rtl {{ direction: rtl; text-align: right; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="timestamp">Generated: {timestamp}</p>

<h2>Summary (Averages per Model)</h2>
{summary.to_html(classes="summary-table")}

<h2>Detailed Results</h2>
{df.to_html(index=False, classes="detail-table")}

</body>
</html>"""

    return html


def export_html_bytes(results: list[MetricsResult], title: str = "ASR Benchmark Report") -> bytes:
    """Export HTML report as bytes (for Streamlit download)."""
    html = export_html_report(results, title)
    return html.encode("utf-8")
