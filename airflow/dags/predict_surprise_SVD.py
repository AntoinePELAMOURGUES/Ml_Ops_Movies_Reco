import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import mlflow
import pickle
from datetime import datetime

# Configuration de MLflow
mlflow.set_tracking_uri("http://mlflow_webserver:5000")
EXPERIMENT_NAME = "Movie_Recommendation_Experiment"
time = datetime.now()
run_name = f"{time}"

def read_ratings(ratings_csv: str, data_dir: str = "/opt/airflow/data/raw") -> pd.DataFrame:
    """Lit le fichier CSV contenant les évaluations des films."""
    try:
        # Lire le fichier CSV et retourner un DataFrame Pandas
        data = pd.read_csv(os.path.join(data_dir, ratings_csv))
        print("Dataset ratings loaded")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def train_model() -> tuple:
    """Entraîne le modèle de recommandation sur les données fournies et retourne le modèle et son RMSE."""
    # Démarrer un nouveau run dans MLflow
    with mlflow.start_run(run_name=run_name) as run:
        # Charger les données d'évaluation des films
        ratings = read_ratings('processed_ratings.csv')

        # Préparer les données pour Surprise
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)

        # Diviser les données en ensembles d'entraînement et de test
        trainset, testset = train_test_split(data, test_size=0.25)

        # Créer et entraîner le modèle SVD
        model = SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)
        model.fit(trainset)

        # Tester le modèle sur l'ensemble de test et calculer RMSE
        predictions = model.test(testset)
        acc = accuracy.rmse(predictions)
        # Arrondir à 2 chiffres après la virgule
        acc_rounded = round(acc, 2)

        print("Valeur de l'écart quadratique moyen (RMSE) :", acc_rounded)

        # Enregistrer les métriques dans MLflow pour suivi ultérieur
        mlflow.log_param("n_factors", 150)
        mlflow.log_param("n_epochs", 30)
        mlflow.log_param("lr_all", 0.01)
        mlflow.log_param("reg_all", 0.05)

        """Récupère le dernier RMSE enregistré dans MLflow."""
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Récupérer les dernières exécutions triées par RMSE décroissant, en prenant la première (meilleure)
        runs = mlflow.search_runs(order_by=["metrics.rmse desc"], max_results=1)
        print("Chargement des anciens RMSE pour comparaison")

        if not runs.empty:
            last_rmse = runs.iloc[0]["metrics.rmse"]
        else:
            last_rmse = float('inf')  # Si aucun run n'est trouvé, retourner une valeur infinie

        print(f"Meilleur RMSE actuel: {last_rmse}, Nouveau RMSE: {acc_rounded}")

        if acc_rounded < last_rmse:
            print("Nouveau modèle meilleur, sauvegarde...")

            directory = '/opt/airflow/model/model_svd.pkl'

            with open(directory, 'wb') as file:
                pickle.dump(model, file)
                print(f'Modèle sauvegardé sous {directory}')
        else:
            print("Ancien modèle conservé ...")
# Définition du DAG Airflow

svd_dag = DAG(
    dag_id='SVD_train_and_compare_model',
    description='SVD Model for Movie Recommendation',
    tags=['reco_movies'],
    schedule_interval='@daily',
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 11, 3),
    }
)

# Tâches du DAG

train_task = PythonOperator(
   task_id='train_model',
   python_callable=train_model,
   dag=svd_dag,
)

if __name__ == "__main__":
   svd_dag.cli()