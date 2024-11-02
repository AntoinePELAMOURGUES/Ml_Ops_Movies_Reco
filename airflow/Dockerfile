FROM apache/airflow:2.9.1

USER root

# Installer gosu
RUN apt-get update && apt-get install -y gosu

USER airflow

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt