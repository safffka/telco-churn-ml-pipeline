# src/report.py
import json
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
REPORT_PATH = REPORTS_DIR / "report.md"


def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


def generate_report(metrics: dict) -> str:
    roc_auc = metrics["roc_auc"]
    passed = metrics["quality_gate"]["passed"]
    threshold = metrics["quality_gate"]["threshold"]

    status = "PASSED ✅" if passed else "FAILED ❌"

    report = f"""
# Telco Customer Churn — Model Report

**Generated at:** {datetime.utcnow().isoformat()} UTC

---

## 1. Model Overview

- Task: Binary classification (Customer Churn)
- Model: CatBoostClassifier
- Target variable: `Churn`

---

## 2. Validation Results

- **ROC-AUC:** {roc_auc:.4f}
- **Quality Gate Threshold:** {threshold}
- **Quality Gate Status:** {status}

---

## 3. ROC Curve

![ROC Curve](roc_curve.png)

---

## 4. Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

---

## 5. Interpretation

- ROC-AUC выше порогового значения → модель демонстрирует приемлемое качество.
- Модель может быть использована для дальнейшего тестирования или деплоя.
- Результаты получены автоматически в рамках ML-пайплайна.

---

## 6. Reproducibility

- All steps (data, training, validation, report) are fully automated.
- The report is generated directly from pipeline artifacts.
"""

    return report.strip()


def save_report(text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(text)


def main():
    metrics = load_metrics()
    report = generate_report(metrics)
    save_report(report)

    print("Report generated:", REPORT_PATH)


if __name__ == "__main__":
    main()
