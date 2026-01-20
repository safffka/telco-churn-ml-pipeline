Telco Customer Churn 
## Описание проекта

Данный проект представляет собой end-to-end ML-пайплайн для задачи прогнозирования оттока клиентов (Customer Churn) на основе датасета Telco Customer Churn.

Пайплайн полностью автоматизирован и реализован в Docker-окружении, без использования локальных виртуальных окружений.
Все этапы — от подготовки данных до валидации, отчёта, трекинга экспериментов и CI — воспроизводимы и запускаются одной командой.

Проект демонстрирует production-подход к ML-разработке.

## Цель

Построить модель, которая предсказывает вероятность ухода клиента (Churn = 1) на основе:

демографических данных

используемых сервисов

контрактных условий

финансовых показателей
 
## Используемые технологии

Python 3.11

Docker

CatBoost (нативная поддержка категориальных признаков)

MLflow (трекинг экспериментов)

GitHub Actions (CI)

Pandas / NumPy / scikit-learn

Matplotlib



## Архитектура пайплайна
Данные → Фичи → Обучение → Валидация → Quality Gate
      → Отчёт → MLflow → CI


Каждый этап:

запускается автоматически

имеет чёткий вход и выход

сохраняет артефакты

## Как запустить проект
 Предварительные требования

     Docker установлен и запущен

     В папке data/raw/ должен лежать файл telco.csv



Сборка Docker-образа

```
docker build -t telco-churn .
```

Этап 1 — Подготовка данных и фичей
```
docker run --rm \
  -v $(pwd)/data:/app/data \
  telco-churn make features
```

Результат:

data/processed/features.parquet

Этап 2 — Обучение модели CatBoost
```
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  telco-churn make train
```

Результат:

models/model.cbm

Этап 3 — Валидация и quality gate
```
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/mlruns:/app/mlruns \
  telco-churn make evaluate
```

Результат:

metrics.json

roc_curve.png

confusion_matrix.png



Если ROC-AUC ниже порога — пайплайн завершается с ошибкой.

Этап 4 — Автоматический отчёт
```
docker run --rm \
  -v $(pwd)/reports:/app/reports \
  telco-churn make report
```

Результат:

reports/report.md

Запуск всего пайплайна одной командой
```
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/mlruns:/app/mlruns \
  telco-churn make all
```
## MLflow UI

Для просмотра экспериментов:
```
docker run --rm -p 5001:5000 \
  -v $(pwd)/mlruns:/app/mlruns \
  telco-churn \
  python -m mlflow ui \
    --host 0.0.0.0 \
    --port 5000
```

Открыть в браузере:

http://localhost:5001

## CI: GitHub Actions

При каждом git push:

собирается Docker-образ

запускается make all

применяется quality gate

сохраняются артефакты (модель, отчёты, MLflow)

Workflow:
.github/workflows/ci.yml

## Метрики

ROC-AUC (основная)

Accuracy

Confusion Matrix

ROC Curve