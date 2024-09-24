import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from fastapi import Request, APIRouter, HTTPException
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
from pydantic import BaseModel
from surprise.prediction_algorithms.matrix_factorization import SVD
from sklearn.preprocessing import LabelEncoder
import re

# Création d'un routeur pour gérer les routes de prédiction
router = APIRouter(
    prefix='/predict',  # Préfixe pour toutes les routes dans ce routeur
    tags=['predict']    # Tag pour la documentation
)

def read_ratings(ratings_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les évaluations des films.

    :param ratings_csv: Nom du fichier CSV contenant les évaluations.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les évaluations.
    """
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings chargé")
    return data

def read_movies(movies_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les informations sur les films.

    :param movies_csv: Nom du fichier CSV contenant les informations sur les films.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les informations sur les films.
    """
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies chargé")
    return df

def read_links(links_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les informations sur les liens des affiches scrappés.

    :param links_csv: Nom du fichier CSV contenant les liens des affiches.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant movieId et lien vers les affiches.
    """
    df = pd.read_csv(os.path.join(data_dir, links_csv))
    df = df[['movieId', 'cover_link']]
    print("Dataset links chargé")
    return df

# Récupération des films les mieux notés par l'utilisateur
def best_user_movies(df, user_id, n =5):
    top_user_movies = (df[df['userId'] == user_id].sort_values(by='rating', ascending=False))[:n]
    top_title = list(top_user_movies['title'])
    top_cover = list(top_user_movies['cover_link'])
    top_genres = list(top_user_movies['genres'])
    return top_title, top_cover, top_genres

# Chargement du dernier modèle
def load_latest_model(directory = "/app/model") -> SVD:
    """Charge le modèle avec la dernière version à partir d'un répertoire."""
    # Vérifier si le répertoire existe
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    # Liste des fichiers dans le répertoire
    files = os.listdir(directory)
    # Filtrer les fichiers pour ne garder que ceux qui correspondent au modèle SVD
    model_files = [f for f in files if f.startswith("model_SVD") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("Aucun modèle SVD trouvé dans le répertoire.")
    # Extraire les numéros de version et trouver le fichier avec la plus grande version
    versioned_files = {}
    for model_file in model_files:
        # Utiliser une expression régulière pour extraire la version du nom de fichier
        match = re.search(r'_(v\d+)', model_file)
        if match:
            version = match.group(1)  # Récupérer la version (ex: v1)
            versioned_files[version] = model_file
    # Trier les versions pour obtenir la dernière
    latest_version = sorted(versioned_files.keys(), key=lambda x: int(x[1:]))[-1]
    latest_model_file = versioned_files[latest_version]
    # Charger le modèle
    filepath = os.path.join(directory, latest_model_file)
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
        print(f'Modèle chargé depuis {filepath} (version {latest_version})')
    return model

def preprocess_data():
    # Chargement des données
    ratings = read_ratings('ratings.csv')
    movies = read_movies('movies.csv')
    links = read_links('links2.csv')
    print("Encodage en cours")
    df = pd.merge(ratings, movies[['movieId', 'title', 'genres']], on = 'movieId', how = 'left')
    df = df.merge(links, on = 'movieId', how= 'left')
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['movieId'] = movie_encoder.fit_transform(df['movieId'])
    return df, movie_encoder, movies, links

# Chargement des données preprocessé
df, movie_encoder, movies, links = preprocess_data()
model_svd = load_latest_model()


def get_top_n_recommendations(user_id: int, n: int = 10) -> List[str]:
    """
    Récupère les n meilleures recommandations de films pour un utilisateur donné.

    :param user_id: L'identifiant de l'utilisateur pour lequel faire des recommandations.
    :param n: Le nombre de recommandations à retourner (par défaut 10).
    :return: Une liste des titres de films recommandés.
    """
    # Obtenir les films déjà notés par l'utilisateur
    user_movies = df[df['userId'] == user_id]['title'].unique()

    # Obtenir les films déjà notés par l'utilisateur
    user_movies = df[df['userId'] == user_id]['movieId'].unique()

    # Obtenir tous les titres de films
    all_movies = df['movieId'].unique()

    # Identifier les films qui n'ont pas été notés par l'utilisateur
    movies_to_predict = list(set(all_movies) - set(user_movies))

    # Créer des paires utilisateur-titre pour prédiction
    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]

    # Prédictions avec le modèle SVD
    predictions_cf = []

    for user_id, title, _ in user_movie_pairs:
        movie_id = df[df['title'] == title]['movieId'].values[0]  # Récupérer movieId à partir du titre
        predicted_rating = model_svd.predict(user_id, movie_id).est  # Utiliser predict avec movieId
        predictions_cf.append((title, predicted_rating))

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    for pred in top_n_recommendations:
        predicted_rating = pred.est

    # Extraire seulement les titres des films recommandés
    top_n_movies = [title for title, _ in top_n_recommendations]

    top_n_movies = movie_encoder.inverse_transform(top_n_movie_ids)
    print({"top 10 movieId" : top_n_movies})
    return top_n_movies

def validate_userId(userId):
    # Vérifier si userId est dans la plage valide
    if userId < 1 or userId > 138493:
        return "Le numéro d'utilisateur doit être compris entre 1 et 138493."

    return None

# Metrics à surveiller avec prometheus
collector = CollectorRegistry()
# Nbre de requête
nb_of_requests_counter = Counter(
    name='predict_nb_of_requests',
    documentation='number of requests per method or per endpoint',
    labelnames=['method', 'endpoint'],
    registry=collector)
# codes de statut des réponses
status_code_counter = Counter(
    name='predict_response_status_codes',
    documentation='Number of HTTP responses by status code',
    labelnames=['status_code'],
    registry=collector)
# Taille des réponses
response_size_histogram = Histogram(
    name='http_response_size_bytes',
    documentation='Size of HTTP responses in bytes',
    labelnames=['method', 'endpoint'],
    registry=collector)
# Temps de traitement par utilisateur
duration_of_requests_histogram = Histogram(
    name='duration_of_requests',
    documentation='Duration of requests per method or endpoint',
    labelnames=['method', 'endpoint', 'user_id'],
    registry=collector)
# Erreurs spécifiques
error_counter = Counter(
    name='api_errors',
    documentation='Count of API errors by type',
    labelnames=['error_type'],
    registry=collector)

# Modèle Pydantic pour la récupération de l'user_id lié aux films
class MovieUserId(BaseModel):
    userId : int  # Nom d'utilisateur

@router.post("/")
async def predict(movie_user_id: MovieUserId) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.
    """
    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/predict').inc()
    # Récupération des données Streamlit
    user_id = movie_user_id.userId
    # valider l'user_id
    userId_error = validate_userId(user_id)
    if userId_error:
        error_counter.labels(error_type='invalid_userId').inc()
        raise HTTPException(status_code=400, detail=userId_error)
    best_user_title, best_user_cover, best_user_genres = best_user_movies(df, user_id)
    recommendations = get_top_n_recommendations(user_id)
    top_n_movies_titles = movies[movies['movieId'].isin(recommendations)]['title'].tolist()
    top_n_movies_cover = links[links['movieId'].isin(recommendations)]['cover_link'].tolist()
    top_n_movies_genres = movies[movies['movieId'].isin(recommendations)]['genres'].tolist()
    # Créer un dictionnaire pour stocker les titres et les couvertures sans doublons
    unique_movies = [{"title" : title, "cover":cover, "genres": genre} for title, cover, genre in zip(top_n_movies_titles, top_n_movies_cover, top_n_movies_genres)]

    best_user_films = [{"title": user_title, "cover": user_cover, "genres" : user_genres}
            for user_title, user_cover, user_genres in zip(best_user_title, best_user_cover, best_user_genres)]

    # Créer le résultat final
    result = {
        "best_user_movies": best_user_films,
        "recommendations": unique_movies
    }
    # Mesurer la taille de la réponse et l'enregistrer
    response_size = len(json.dumps(result))
    # Calculer la durée et enregistrer dans l'histogramme
    duration = time.time() - start_time
    # Enregistrement des métriques
    status_code_counter.labels(status_code = "200").inc()
    duration_of_requests_histogram.labels(method='POST', endpoint='/predict', user_id=str(user_id)).observe(duration)
    response_size_histogram.labels(method='POST', endpoint='/predict').observe(response_size)
    print({"result_request_fastapi" : result})
    return result


