from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

REFERENCE = Path("data/monitoring/reference.parquet")
CURRENT = Path("data/monitoring/current.parquet")
OUT_HTML = Path("reports/monitoring_drift.html")

MAX_SHARE_DRIFTED = 0.30  # 30% фичей с дрейфом = тревога

def main():
    if not REFERENCE.exists():
        raise FileNotFoundError(f"Reference not found: {REFERENCE}")
    if not CURRENT.exists():
        raise FileNotFoundError(f"Current not found: {CURRENT}")

    ref = pd.read_parquet(REFERENCE)
    cur = pd.read_parquet(CURRENT)

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(reference_data=ref, current_data=cur)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(OUT_HTML))

    summary = report.as_dict()

    # DataDriftPreset — это 0-й элемент metrics
    drift_share = summary["metrics"][0]["result"]["share_of_drifted_columns"]
    print({"share_of_drifted_columns": drift_share})

    if drift_share > MAX_SHARE_DRIFTED:
        raise RuntimeError(
            f"Data drift too high: {drift_share:.2f} > {MAX_SHARE_DRIFTED}"
        )

    print(f"Monitoring report saved: {OUT_HTML}")

if __name__ == "__main__":
    main()
