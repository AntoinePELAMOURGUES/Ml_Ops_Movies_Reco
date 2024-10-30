import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
import mlflow
import mlflow.sklearn
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
def connect_to_mongodb(uri: str, db_name: str):
    """Établit une connexion à la base de données MongoDB."""
    client = MongoClient(uri)
    return client[db_name]

def read_ratings(db) -> pd.DataFrame:
    """Lit les évaluations des films depuis MongoDB."""
    ratings_collection = db.ratings
    data = pd.DataFrame(list(ratings_collection.find()))
    print("Dataset ratings chargé depuis MongoDB")
    return data

def read_movies(db) -> pd.DataFrame:
    """Lit les informations sur les films depuis MongoDB."""
    movies_collection = db.movies
    df = pd.DataFrame(list(movies_collection.find()))
    print("Dataset movies chargé depuis MongoDB")
    return df

def create_X(df):
    """Génère une matrice creuse d'évaluations utilisateur-film."""
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

    return X

def train_model(X, k=10):
    """Entraîne le modèle KNN sur les données d'entraînement."""
    X = X.T  # Transposer pour que les utilisateurs soient en lignes
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric='cosine')

    model = kNN.fit(X)

    return model

def get_predictions(model, X_test):
    """Fait des prédictions sur l'ensemble de test en utilisant le modèle KNN."""
    distances, indices = model.kneighbors(X_test.T)  # Transposer X_test pour obtenir les voisins
    return indices

def calculate_metrics(actual_ratings, predicted_ratings):
    """Calcule MAE et RMSE à partir des évaluations réelles et prédites."""
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    rmse = root_mean_squared_error(actual_ratings, predicted_ratings, squared=False)

    return mae, rmse

def save_model(model, experiment_name: str) -> None:
    """Sauvegarde le modèle entraîné dans MLflow."""
    mlflow.set_experiment(experiment_name)  # Définir l'expérience

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        print(f'Modèle sauvegardé sous l\'ID de run : {run.info.run_id}')

def run_training_and_evaluation(**kwargs):
    """Fonction principale pour entraîner le modèle et évaluer ses performances."""

    # Connexion à MongoDB
    mongo_uri = "mongodb://antoine:pela@mongodb:27017/"
    db_name = "reco_movies"

    db = connect_to_mongodb(mongo_uri, db_name)

    # Chargement des données depuis MongoDB
    ratings = read_ratings(db)

    # Créer la matrice creuse à partir des évaluations
    X = create_X(ratings)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Entraîner le modèle KNN
    model_knn = train_model(X_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred_indices = get_predictions(model_knn, X_test)

    # Calculer les évaluations prédites à partir des indices des voisins
    predicted_ratings = []

    for user_idx in range(y_pred_indices.shape[0]):
        neighbor_ratings = []

        for neighbor_idx in y_pred_indices[user_idx]:
            movie_ids = ratings[ratings['userId'] == neighbor_idx]['movieId'].values

            for movie_id in movie_ids:
                rating = ratings[(ratings['userId'] == neighbor_idx) & (ratings['movieId'] == movie_id)]['rating'].values

                if rating.size > 0:
                    neighbor_ratings.append(rating[0])

        if neighbor_ratings:
            predicted_ratings.append(np.mean(neighbor_ratings))
        else:
            predicted_ratings.append(0)

   # Obtenir les évaluations réelles correspondantes dans l'ensemble de test
    actual_ratings = []
    for user_idx in range(X_test.shape[0]):
       actual_user_id = ratings.iloc[user_idx]['userId']
       actual_movie_id = ratings.iloc[user_idx]['movieId']

       actual_rating = ratings[(ratings['userId'] == actual_user_id) & (ratings['movieId'] == actual_movie_id)]['rating'].values

       if actual_rating.size > 0:
           actual_ratings.append(actual_rating[0])

    # Calculer MAE et RMSE
    mae, rmse = calculate_metrics(actual_ratings, predicted_ratings)
    print(f"MAE: {mae}, RMSE: {rmse}")

    # Sauvegarder le modèle dans MLflow
    save_model(model_knn, 'MovieRecommendationModel')

# Définir le DAG Airflow
my_dag = DAG(
   dag_id='KNN_train_model',
   description='Modèle KNN pour la recommandation de films',
   tags=['reco_movies'],
   schedule_interval='@daily',
   default_args={
       'owner': 'airflow',
       'start_date': datetime(2024, 10, 30),
   }
)

# Créer une tâche pour entraîner le modèle et évaluer ses performances
train_task = PythonOperator(
   task_id='train_and_evaluate_model',
   python_callable=run_training_and_evaluation,
   dag=my_dag,
)

if __name__ == "__main__":
   my_dag.cli()