import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import pickle
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
import mlflow
import mlflow.sklearn  # Assurez-vous d'importer mlflow.sklearn

# Configuration de MLflow
mlflow.set_tracking_uri("http://mlflow-webserver:5000")
mlflow.set_experiment("Movie_Recommendation_Experiment")

def read_ratings(ratings_csv: str, data_dir: str = "/opt/airflow/data/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie ratings."""
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings loaded")
    return data

def train_model(df: pd.DataFrame) -> SVD:
    """Entraîne le modèle de recommandation sur les données fournies."""
    # Démarrer un nouveau run dans MLflow
    with mlflow.start_run() as run:
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader=reader)
        trainset = data.build_full_trainset()

        model = SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)
        model.fit(trainset)

        print("Début de la cross-validation")
        cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)

        mean_rmse = cv_results['test_rmse'].mean()
        print("Moyenne des RMSE :", mean_rmse)

        # Enregistrer les métriques dans MLflow
        mlflow.log_param("n_factors", 150)
        mlflow.log_param("n_epochs", 30)
        mlflow.log_param("lr_all", 0.01)
        mlflow.log_param("reg_all", 0.05)
        mlflow.log_metric("mean_rmse", mean_rmse)

        # Sauvegarder le modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")

    return model, mean_rmse

def save_model(model: SVD, filepath: str, version: str) -> None:
    """Sauvegarde le modèle entraîné dans un fichier."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    base, ext = os.path.splitext(filepath)
    versioned_filepath = f"{base}_{version}{ext}"

    with open(versioned_filepath, 'wb') as file:
        pickle.dump(model, file)
        print(f'Modèle sauvegardé sous {versioned_filepath}')

# Définition du DAG Airflow
default_args = {
    'owner': 'airflow',
    'start_date': pd.to_datetime('2024-11-01'),
}

dag = DAG(
    'movie_recommendation_dag',
    tag=['reco_movies'],
    default_args=default_args,
    description='DAG for training movie recommendation model with MLflow integration',
    schedule_interval='@daily',
)

# Tâches du DAG
read_ratings_task = PythonOperator(
    task_id='read_ratings',
    python_callable=read_ratings,
    op_kwargs={'ratings_csv': 'ratings.csv'},
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs={'df': "{{ task_instance.xcom_pull(task_ids='read_ratings') }}"},
    dag=dag,
)

save_model_task = PythonOperator(
    task_id='save_model',
    python_callable=save_model,
    op_kwargs={'model': "{{ task_instance.xcom_pull(task_ids='train_model') }}", 'filepath': '/opt/airflow/models/model_SVD.pkl', 'version': 'v1'},
    dag=dag,
)

# Définir l'ordre des tâches
read_ratings_task >> train_model_task >> save_model_task