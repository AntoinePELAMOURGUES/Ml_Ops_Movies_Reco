import pandas as pd
import os
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def read_ratings(ratings_csv: str, data_dir: str = "/app/data") -> pd.DataFrame:
    """Lit le fichier CSV contenant les évaluations des films."""
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings chargé")
    return data

def read_movies(movies_csv: str, data_dir: str = "/app/data") -> pd.DataFrame:
    """Lit le fichier CSV contenant les informations sur les films."""
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies chargé")
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

def train_model(X, k = 10):
    X = X.T
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric='cosine')
    model = kNN.fit(X)
    return model

def save_model(model, filepath: str) -> None:
    """Sauvegarde le modèle entraîné dans un fichier."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    directory = os.path.join(filepath, 'model_knn.pkl')
    with open(directory, 'wb') as file:
        pickle.dump(model, file)
        print(f'Modèle sauvegardé sous {filepath}/model.pkl')


def read_version(filepath: str) -> int:
    """Lit la version actuelle à partir d'un fichier."""
    if not os.path.exists(filepath):
        return 0  # Valeur par défaut si le fichier n'existe pas
    with open(filepath, 'r') as file:
        return int(file.read().strip())

def write_version(filepath: str, version: int) -> None:
    """Écrit la nouvelle version dans un fichier."""
    with open(filepath, 'w') as file:
        file.write(str(version))

if __name__ == "__main__":

    # Chargement des données
    ratings = read_ratings('ratings.csv')
    movies = read_movies('movies.csv')
    X = create_X(ratings)
    model_knn = train_model(X)
    save_model(model_knn, '/app/model')