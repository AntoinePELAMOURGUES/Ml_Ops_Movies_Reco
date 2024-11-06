FROM apache/airflow:latest

USER root

# Installer les dépendances nécessaires pour la compilation et gosu
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gosu \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .

# Mettre à jour pip et installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt