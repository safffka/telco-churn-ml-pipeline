## Telco Customer Churn 


Отток клиентов — один из главных источников потерь выручки в телеком-бизнесе. 
Даже снижение churn на 1–2% может давать миллионы экономии за счёт роста LTV и снижения затрат на привлечение новых клиентов.

Этот проект решает задачу **раннего выявления клиентов с высоким риском оттока** и показывает, 
как ML-модель может быть встроена в production-процесс:
от данных и обучения до контроля качества и мониторинга после деплоя.

Результат работы модели — вероятность ухода клиента, которая может быть напрямую использована 
в retention-кампаниях (скидки, персональные предложения, изменение тарифов).
## Цель

Построить модель, которая предсказывает вероятность ухода клиента (Churn = 1) на основе:

демографических данных,

используемых сервисов,

контрактных условий,

финансовых показателей.

Результат модели — вероятность оттока, которая может быть использована бизнесом для таргетированных retention-действий.
 
## Используемые технологии

Python 3.11

Docker

CatBoost (нативная поддержка категориальных признаков)

MLflow (трекинг экспериментов и артефактов)

Evidently (post-deploy monitoring и data drift)

GitHub Actions (CI)

Pandas / NumPy / scikit-learn

Matplotlib

Pytest



## Архитектура пайплайна
``` 
Raw Data
   ↓
Feature Engineering
   ↓
Model Training (CatBoost)
   ↓
Validation & Quality Gate
   ↓
Auto Report
   ↓
MLflow Tracking
   ↓
CI
   ↓
Post-deploy Monitoring (Evidently)
```


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
## Post-deploy monitoring
После обучения и деплоя модели реализован мониторинг качества данных и предсказаний.

Мониторинг включает:

data drift по входным признакам,

prediction drift по выходу модели,

HTML-отчёт,

quality gate по доле дрейфующих признаков.


Генерация reference и current данных
```
docker run --rm -v $(pwd):/app telco-churn python scripts/make_reference.py
docker run --rm -v $(pwd):/app telco-churn python scripts/make_current.py
```
Запуск мониторинга
```
docker run --rm -v $(pwd):/app telco-churn make monitor
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