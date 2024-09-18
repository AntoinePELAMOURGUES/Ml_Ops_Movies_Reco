import pandas as pd
import os
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

def read_ratings(ratings_csv: str, data_dir: str = "/home/antoine/Ml_Ops_Movies_Reco/src/data/data/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV des évaluations et retourne un DataFrame.

    :param ratings_csv: Nom du fichier CSV contenant les évaluations.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les évaluations.
    """
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    return data


def read_movies(movies_csv: str, data_dir: str = "/home/antoine/Ml_Ops_Movies_Reco/src/data/data/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV des films, encode les titres et retourne un DataFrame.

    :param movies_csv: Nom du fichier CSV contenant les films.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les films avec les titres encodés.
    """
    # Lire le fichier CSV
    df = pd.read_csv(os.path.join(data_dir, movies_csv))

    # Initialiser LabelEncoder
    le = LabelEncoder()

    # Ajuster et transformer la colonne des titres
    df['title'] = le.fit_transform(df['title'])

    return df


def create_user_matrix(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice utilisateur à partir des évaluations et des films.

    :param ratings: DataFrame contenant les évaluations des utilisateurs.
    :param movies: DataFrame contenant les films.
    :return: DataFrame contenant la matrice utilisateur.
    """
    # Fusionner les deux tables
    movie_ratings = ratings.merge(movies, on="movieId", how="left")

    # Sélectionner les colonnes pertinentes
    movie_ratings = movie_ratings[['userId', 'title', 'rating']]

    return movie_ratings


if __name__ == "__main__":
    # Lire les évaluations des utilisateurs et les films
    user_ratings = read_ratings("ratings.csv")
    movies = read_movies("movies.csv")

    # Créer la matrice utilisateur
    user_matrix = create_user_matrix(user_ratings, movies)

    # Définir le chemin de sauvegarde
    filepath = "/home/antoine/Ml_Ops_Movies_Reco/src/data/processed"
    matrix_file = os.path.join(filepath, "mv_matrix_surprise.csv")

    # Vérifier si le dossier existe
    if not os.path.exists(filepath):
        # Si le dossier n'existe pas, le créer
        os.makedirs(filepath)

    # Sauvegarder la matrice utilisateur dans un fichier CSV
    user_matrix.to_csv(matrix_file, index=False)
    print(f"Fichier matrix_file sauvegardé dans {matrix_file}")
