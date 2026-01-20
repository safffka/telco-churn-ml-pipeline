FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# рабочая директория
WORKDIR /app

# зависимости — раньше кода (кэш Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код и данные
COPY src ./src
COPY data ./data
COPY Makefile .


CMD ["make", "train"]

