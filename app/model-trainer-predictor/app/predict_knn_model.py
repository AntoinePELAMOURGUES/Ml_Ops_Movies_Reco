import pandas as pd
import os
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
import mlflow
import mlflow.sklearn

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
    """
    Génère une matrice creuse avec quatre dictionnaires de mappage
    - user_mapper: mappe l'ID utilisateur à l'index utilisateur
    - movie_mapper: mappe l'ID du film à l'index du film
    - user_inv_mapper: mappe l'index utilisateur à l'ID utilisateur
    - movie_inv_mapper: mappe l'index du film à l'ID du film
    Args:
        df: pandas dataframe contenant 3 colonnes (userId, movieId, rating)

    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X

def train_model(X, k=10):
    X = X.T
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric='cosine')
    model = kNN.fit(X)
    return model

def save_model(model, experiment_name: str) -> None:
    """Sauvegarde le modèle entraîné dans MLflow."""

    mlflow.set_experiment(experiment_name)  # Définir l'expérience

    with mlflow.start_run() as run:
        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")
        print(f'Modèle sauvegardé sous l\'ID de run : {run.info.run_id}')


if __name__ == "__main__":
    # Connexion à MongoDB
    mongo_uri = "mongodb://antoine:pela@mongodb:27017/"
    db_name = "reco_movies"

    db = connect_to_mongodb(mongo_uri, db_name)

    # Chargement des données depuis MongoDB
    ratings = read_ratings(db)
    movies = read_movies(db)

    X = create_X(ratings)
    model_knn = train_model(X)

    # Enregistrer le modèle avec un nom d'expérience spécifique
    save_model(model_knn, 'MovieRecommendationModel')