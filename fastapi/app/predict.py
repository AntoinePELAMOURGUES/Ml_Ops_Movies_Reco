import pandas as pd
import os
import json
import pickle
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rapidfuzz import process, fuzz, utils
from fastapi import Request, APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
from pydantic import BaseModel

# ROUTEUR POUR GERER LES ROUTES PREDICT

router = APIRouter(
    prefix='/predict',  # Préfixe pour toutes les routes dans ce routeur
    tags=['predict']    # Tag pour la documentation
)

# ---------------------------------------------------------------

# ENSEMBLE DES FONCTIONS UTILISEES

# Chargement des datasets
def read_ratings(ratings_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie ratings."""
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings loaded")
    return data

def read_movies(movies_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie information."""
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies loaded")
    return df

def read_links(links_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie information."""
    df = pd.read_csv(os.path.join(data_dir, links_csv))
    print("Dataset links loaded")
    return df


# Chargement du dernier modèle
def load_model(pkl_files, directory = "/app/model") :
    """Charge le modèle à partir d'un répertoire."""
    # Vérifier si le répertoire existe
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    # Charger le modèle
    filepath = os.path.join(directory, pkl_files)
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
        print(f'Modèle chargé depuis {filepath}')
    return model

# Fonction pour obtenir des recommandations
def get_recommendations(user_id: int, model: SVD, ratings_df: pd.DataFrame, n_recommendations: int = 10):
    """Obtenir des recommandations pour un utilisateur donné."""
    # Créer un DataFrame contenant tous les films
    all_movies = ratings_df['movieId'].unique()

    # Obtenir les films déjà évalués par l'utilisateur
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()

    # Trouver les films non évalués par l'utilisateur
    unseen_movies = [movie for movie in all_movies if movie not in rated_movies]

    # Préparer les prédictions pour les films non évalués
    predictions = []
    for movie_id in unseen_movies:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))  # Ajouter l'ID du film et la note prédite

    # Trier les prédictions par note prédite (descendant) et prendre les meilleures n_recommendations
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]

    return top_n  # Retourner les meilleures recommandations



# Fonction recommendation si utilisateur inconnu
def get_content_based_recommendations(cosine_sim, title, n_recommendations=10):
    # Récupérer l'ID du film à partir du titre
    idx = movie_idx[title]
    # Trouver l'index correspondant dans la matrice de similarité
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Trier les films par score de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Obtenir les n recommandations (en ignorant le premier qui est le film lui-même)
    sim_scores = sim_scores[1:(n_recommendations + 1)]
    # Récupérer les indices des films similaires
    similar_movies_indices = [i[0] for i in sim_scores]
    # Récupérer les movieId correspondants
    similar_movies_ids = movies['movieId'].iloc[similar_movies_indices].tolist()
    return similar_movies_ids  # Retourner les IDs des films similaires

# Validation de l'utilisateur
def validate_userId(userId):
    # Vérifier si userId est dans la plage valide
    if userId < 1 or userId > 138493:
        return "Le numéro d'utilisateur doit être compris entre 1 et 138493."
    return None

# Recherche un titre proche de la requete
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]


# ---------------------------------------------------------------

# METRICS PROMETHEUS

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

# ---------------------------------------------------------------

# CHARGEMENT DES DONNEES AU DEMARRAGE DE API
print("DEBUT DES CHARGEMENTS")
# Chargement de nos dataframe depuis mongo_db
ratings = read_ratings('processed_ratings.csv')
movies = read_movies('processed_movies.csv')
links = read_links('processed_links.csv')
# Chargement d'un modèle SVD pré-entraîné pour les recommandations
model_svd = load_model('model_svd.pkl')
# Chargement de la matrice cosinus similarity
cosine_sim = load_model('cosinus_similarity.pk')
print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")
# Merge de movies et links pour avoir un ix commun
movies_links_df = movies.merge(links, on = "movieId", how = 'left')
# Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
movie_idx = dict(zip(movies['title'], list(movies.index)))
cover_idx = dict(zip(movies_links_df['cover_link'], list(movies_links_df.index)))
# Création de dictionnaires pour accéder facilement aux titres et aux couvertures des films par leur ID
movie_titles = dict(zip(movies['movieId'], movies['title']))
movie_covers = dict(zip(links['movieId'], links['cover_link']))

print("FIN DES CHARGEMENTS")
# ---------------------------------------------------------------

# REQUETES API

# Modèle Pydantic pour la récupération de l'user_id lié aux films
class UserRequest(BaseModel):
    userId: Optional[int] = None  # Nom d'utilisateur
    movie_title : str # Nom du film

# Route API concernant les utilisateurs déjà identifiés
@router.post("/identified_user")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.
    Args: user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur, y compris l'ID utilisateur et le titre du film.
    Returns:
        Dict[str, Any]: Un dictionnaire contenant le choix de l'utilisateur et les recommandations de films.
    """
    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/identified_user').inc()
    # Récupération des données Streamlit
    print({"user_request" : user_request})
    movie_title = movie_finder(user_request.movie_title)  # Trouver le titre du film correspondant
    user_id = user_request.userId  # Récupérer l'ID utilisateur depuis la requête
    # Validation de l'ID utilisateur
    userId_error = validate_userId(user_id)
    if userId_error:
        error_counter.labels(error_type='invalid_userId').inc()  # Incrémenter le compteur d'erreurs
        raise HTTPException(status_code=400, detail=userId_error)  # Lever une exception si l'ID est invalide
    # Récupération de l'ID du film à partir du titre
    movie_id = int(movies['movieId'][movies['title'] == movie_title].iloc[0])
    # Récupérer les ID des films recommandés en utilisant la fonction de similarité
    recommendations = get_recommendations(user_id, model_svd, ratings)
    # Obtenir le titre et la couverture du film choisi par l'utilisateur
    movie_title = movie_titles[movie_id]
    movie_cover = movie_covers[movie_id]
    # Créer un dictionnaire pour stocker les titres et les couvertures des films recommandés
    result = {
        "user_choice": {"title": movie_title, "cover": movie_cover},
        "recommendations": [{"title": movie_titles[i], "cover": movie_covers[i]} for i in recommendations]
    }
    # Mesurer la taille de la réponse et l'enregistrer
    response_size = len(json.dumps(result))
    # Calculer la durée et enregistrer dans l'histogramme
    duration = time.time() - start_time
    # Enregistrement des métriques pour Prometheus
    status_code_counter.labels(status_code="200").inc()  # Compter les réponses réussies
    duration_of_requests_histogram.labels(method='POST', endpoint='/identified_user', user_id=str(user_id)).observe(duration)  # Enregistrer la durée de la requête
    response_size_histogram.labels(method='POST', endpoint='/identified_user').observe(response_size)  # Enregistrer la taille de la réponse
    print({"result_request_fastapi": result})  # Afficher le résultat dans la console pour le débogage
    return result

# Route Api concernant les nouveaux utilisateurs
@router.post("/new_user")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur le contenu. .
    Args: user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur, ici seulement le titre du film.
    Returns:
        Dict[str, Any]: Un dictionnaire contenant le choix de l'utilisateur et les recommandations de films.
    """
    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/new_user').inc()
    # Récupération des données Streamlit
    movie_title = movie_finder(user_request.movie_title)  # Trouver le titre du film correspondant
    # Récupération de l'ID du film à partir du titre
    movie_id = int(movies['movieId'][movies['title'] == movie_title].iloc[0])
    similar_movies = get_content_based_recommendations(cosine_sim, movie_title, n_recommendations=10)
    # Obtenir le titre et la couverture du film choisi par l'utilisateur
    user_movie_title = movie_titles[movie_id]
    user_movie_cover = movie_covers[movie_id]
    # Créer un dictionnaire pour stocker les titres et les couvertures des films recommandés
    result = {
        "user_choice": {"title": user_movie_title, "cover": user_movie_cover},
        "recommendations": [{"title": movie_titles[i], "cover": movie_covers[i]} for i in similar_movies]}
    # Mesurer la taille de la réponse et l'enregistrer
    response_size = len(json.dumps(result))
    # Calculer la durée et enregistrer dans l'histogramme
    duration = time.time() - start_time
    # Enregistrement des métriques pour Prometheus
    status_code_counter.labels(status_code="200").inc()  # Compter les réponses réussies
    duration_of_requests_histogram.labels(method='POST', endpoint='/new_user', user_id= None).observe(duration)  # Enregistrer la durée de la requête
    response_size_histogram.labels(method='POST', endpoint='/new_user').observe(response_size)  # Enregistrer la taille de la réponse
    print({"result_request_fastapi": result})  # Afficher le résultat dans la console pour le débogage
    return result