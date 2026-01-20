# Telco Customer Churn — Model Report

**Generated at:** 2026-01-20T07:40:52.106667 UTC

---

## 1. Model Overview

- Task: Binary classification (Customer Churn)
- Model: CatBoostClassifier
- Target variable: `Churn`

---

## 2. Validation Results

- **ROC-AUC:** 0.8390
- **Quality Gate Threshold:** 0.75
- **Quality Gate Status:** PASSED ✅

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