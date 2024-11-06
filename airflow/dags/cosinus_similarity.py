from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import mlflow
import pickle
import os

# Configuration de MLflow
mlflow.set_tracking_uri("http://mlflow_webserver:5000")
EXPERIMENT_NAME = "Movie_Recommendation_Experiment"
time = datetime.now()
run_name = f"{time}"

def read_movies(movies_csv: str, data_dir: str = "/opt/airflow/data/raw") -> pd.DataFrame:
    """Lit le fichier CSV."""
    try:
        # Lire le fichier CSV et retourner un DataFrame Pandas
        data = pd.read_csv(os.path.join(data_dir, movies_csv))
        print("Dataset ratings loaded")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_cosine_similarity():
    with mlflow.start_run(run_name=run_name) as run:
        # Charger vos données depuis les fichiers CSV
        movies = read_movies('processed_movies.csv')

        # Prétraitement des genres
        movies['genres'] = movies['genres'].str.replace("'", "").str.strip("[]")
        movies['genres'] = movies['genres'].str.split(', ')

        # Création d'une représentation binaire des genres pour chaque film
        exploded_df = movies.explode('genres')
        genre_dummies = pd.get_dummies(exploded_df['genres'])
        genre_dummies = genre_dummies.groupby(exploded_df['movieId']).sum()

        # Réinitialiser l'index pour fusionner avec le DataFrame d'origine
        genre_dummies.reset_index(inplace=True)
        result_df = pd.merge(movies[['movieId', 'title', 'year']], genre_dummies, on='movieId')

        # Normalisation de l'année
        result_df['year'] = result_df['year'].astype(int)
        result_df['year_normalized'] = (result_df['year'] - result_df['year'].min()) / (result_df['year'].max() - result_df['year'].min())

        # Création des features pour la similarité cosinus
        movie_features = result_df.drop(columns=['title', 'year', '(no genres listed)'])

        # Calcul de la similarité cosinus
        cosine_sim = cosine_similarity(movie_features.drop(columns=['movieId']), movie_features.drop(columns=['movieId']))

        directory = '/opt/airflow/model/cosinus_similarity.pkl'
        # Sauvegarder la matrice dans un fichier Pickle
        with open(directory, 'wb') as file:
            pickle.dump(cosine_sim, file)
            print(f'Matrice sauvegardée sous {directory}')

# Définir le DAG
with DAG('cosine_similarity_dag',
         description='Matrice pour démarrage à froid - user inconnu',
         tags=['reco_movies'],
         default_args={'owner': 'airflow', 'start_date': datetime(2024, 11, 6)},
         schedule_interval='@daily',
         ) as dag:

    create_cosine_similarity_task = PythonOperator(
        task_id='create_cosine_similarity',
        python_callable=create_cosine_similarity,
    )

create_cosine_similarity_task