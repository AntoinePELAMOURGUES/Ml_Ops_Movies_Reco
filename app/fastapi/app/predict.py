import pandas as pd
import os
import pickle
import numpy as np
from surprise import Reader, Dataset, accuracy, SVD
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from fastapi import Request, UploadFile, APIRouter
from typing import List, Dict, Any

# Création d'un routeur pour gérer les routes de prédiction
router = APIRouter(
    prefix='/predict',  # Préfixe pour toutes les routes dans ce routeur
    tags=['predict']    # Tag pour la documentation
)

def read_ratings(ratings_csv: str, data_dir: str = "/code/app/data/") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les évaluations des films.

    :param ratings_csv: Nom du fichier CSV contenant les évaluations.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les évaluations.
    """
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings chargé")
    return data

def read_movies(movies_csv: str, data_dir: str = "/code/app/data/") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les informations sur les films.

    :param movies_csv: Nom du fichier CSV contenant les informations sur les films.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les informations sur les films.
    """
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies chargé")
    return df

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
mlb = MultiLabelBinarizer()

def create_user_matrix(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice utilisateur à partir des évaluations et des informations sur les films.

    :param ratings: DataFrame contenant les évaluations.
    :param movies: DataFrame contenant les informations sur les films.
    :return: DataFrame fusionné et prétraité.
    """
    # Fusionner les évaluations et les informations sur les films
    df = ratings.merge(movies[['movieId', 'genres']], on="movieId", how="left")

    # Encoder userId et movieId
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['movieId'] = movie_encoder.fit_transform(df['movieId'])

    # Traitement des genres
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres').str.split('|')),
                               columns=mlb.classes_, index=df.index))

    # Supprimer la colonne "(no genres listed)" si elle existe
    df = df.drop("(no genres listed)", axis=1, errors='ignore')

    print("Dataset fusionnés et prétraités")
    return df

# Chargement des données
ratings = read_ratings('ratings.csv')
movies = read_movies('movies.csv')
df = create_user_matrix(ratings, movies)

# Chemin du fichier PKL
file_path = '/code/app/model/model_SVD_1.pkl'

# Charger le modèle SVD depuis le fichier PKL
with open(file_path, 'rb') as file:
    model_svd = pickle.load(file)

def get_top_n_recommendations(user_id: int, n: int = 10) -> List[str]:
    """
    Récupère les n meilleures recommandations de films pour un utilisateur donné.

    :param user_id: L'identifiant de l'utilisateur pour lequel faire des recommandations.
    :param n: Le nombre de recommandations à retourner (par défaut 10).
    :return: Une liste des titres de films recommandés.
    """

    user_movies = df[df['userId'] == user_id]['movieId'].unique()

    all_movies = df['movieId'].unique()

    # Identifier les films qui n'ont pas été notés par l'utilisateur
    movies_to_predict = list(set(all_movies) - set(user_movies))

    # Créer des paires utilisateur-film pour prédiction
    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]

    # Prédictions avec le modèle SVD
    predictions_cf = model_svd.test(user_movie_pairs)

    # Trier par note prédite et récupérer les meilleurs n résultats
    top_n_recommendations = sorted(predictions_cf, key=lambda x: x.est, reverse=True)[:n]

    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    # Décoder les IDs de film en titres
    top_n_movies = movie_encoder.inverse_transform(top_n_movie_ids)

    return top_n_movies.tolist()

# Route pour récupérer les tops 10 de notre modèle
@router.post("/")
async def predict(request: Request) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.

    :param request: La requête HTTP contenant l'ID utilisateur.
    :return: Un dictionnaire avec les recommandations de films.
    """

    # Récupération des données Streamlit
    try:
        request_data = await request.form()
        print({'request_data': request_data})
    except ValueError:
        request_data = None

    user_id = int(request_data['userId'])  # Convertir en entier

    recommendations = get_top_n_recommendations(user_id)

    top_n_movies_titles = movies[movies['movieId'].isin(recommendations)]['title'].tolist()

    result = {str(i): title for i, title in enumerate(top_n_movies_titles, 1)}

    return result