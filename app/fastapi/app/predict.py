import pandas as pd
import os
import json
import pickle
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

# Ouverture fichier ratings
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

# Ouverture fichier movies
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

# Ouverture fichier links
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

# Chargement du dernier modèle
def load_model(directory = "/app/model") :
    """Charge le modèle à partir d'un répertoire."""
    # Vérifier si le répertoire existe
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    # Charger le modèle
    filepath = os.path.join(directory, 'model_knn.pkl')
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
        print(f'Modèle chargé depuis {filepath}')
    return model

# Creation de notre matrice creuse
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
      # Nombre unique d'utilisateurs et de films
    M = df['userId'].nunique()  # Compte le nombre d'utilisateurs uniques
    N = df['movieId'].nunique()  # Compte le nombre de films uniques
    # Créer un dictionnaire pour mapper les IDs utilisateurs à des indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    # Créer un dictionnaire pour mapper les IDs de films à des indices
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))
    # Créer un dictionnaire inverse pour mapper les indices utilisateurs aux IDs utilisateurs
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    # Créer un dictionnaire inverse pour mapper les indices de films aux IDs de films
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))
    # Obtenir les indices correspondants pour chaque utilisateur et film dans le DataFrame
    user_index = [user_mapper[i] for i in df['userId']]  # Convertir les IDs utilisateurs en indices
    item_index = [movie_mapper[i] for i in df['movieId']]  # Convertir les IDs de films en indices
    # Créer une matrice creuse en utilisant les évaluations, les indices d'utilisateur et de film
    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Predictions si utilisateur connu
def get_recommendations(user_id: int, model, X, user_mapper, movie_inv_mapper, n_recommendations=10):
    """
    Effectue des recommandations de films pour un utilisateur donné.

    Args:
        user_id: ID de l'utilisateur pour lequel faire des recommandations.
        model: Le modèle KNN chargé.
        X: Matrice creuse contenant les évaluations.
        user_mapper: Dictionnaire qui mappe les IDs utilisateurs aux indices utilisateurs.
        movie_inv_mapper: Dictionnaire qui mappe les indices de films aux IDs de films.
        n_recommendations: Nombre de recommandations à retourner.

    Returns:
        recommendations: Liste des IDs de films recommandés.
    """
    X = X.T

    # Vérifier si l'utilisateur existe dans le mappage
    if user_id not in user_mapper:
        print(f"L'utilisateur {user_id} n'existe pas dans les données.")
        return []

    # Obtenir l'index utilisateur
    user_index = user_mapper[user_id]

    # Obtenir les voisins
    distances, indices = model.kneighbors(X[user_index], n_neighbors=n_recommendations + 1)

    # Exclure l'utilisateur lui-même (premier voisin)
    recommended_indices = indices.flatten()[1:]  # Ignorer le premier voisin (l'utilisateur lui-même)

    # Convertir les indices recommandés en IDs de films
    recommendations = [movie_inv_mapper[i] for i in recommended_indices]

    return recommendations

# Fonction recommendation si utilisateur inconnu
def get_content_based_recommendations(title, n_recommendations=10):
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    return similar_movies # return les index des films similaires

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

# retourne la clé associée à une valeur donnée
def find_key(value, dictionary):
    for key, val in dictionary.items():
        if val == value:
            return key

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
# Lecture des fichiers CSV contenant les données
ratings = read_ratings('ratings.csv')
movies = read_movies('movies.csv')
# Convertit la chaîne de genres en liste en séparant par '|'
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
# Charge les liens vers les couvertures des films
links = read_links('links2.csv')
# Merge de movies et links pour avoir un ix commun
movies_links_df = movies.merge(links, on = "movieId", how = 'left')
# Chargement d'un modèle KNN (K-Nearest Neighbors) pré-entraîné pour les recommandations
model_knn = load_model()
# Création de la matrice X et des mappers pour les utilisateurs et les films
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
# Création d'une représentation binaire des genres pour chaque film
genres = set(g for G in movies['genres'] for g in G)
for g in genres:
    # Ajoute une colonne binaire pour chaque genre (1 si présent, sinon 0)
    movies[g] = movies.genres.transform(lambda x: int(g in x))
# Préparation d'une matrice contenant uniquement les colonnes de genres, sans ID et titre
movie_genres = movies.drop(columns=['movieId', 'title', 'genres'])
# Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
movie_idx = dict(zip(movies['title'], list(movies.index)))
cover_idx = dict(zip(movies_links_df['cover_link'], list(movies_links_df.index)))
# Calcul de la similarité cosinus entre les genres des films
cosine_sim = cosine_similarity(movie_genres, movie_genres)
print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")
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
    recommendations = get_recommendations(movie_id, model_knn, X, user_mapper, movie_inv_mapper)
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
    similar_movies = get_content_based_recommendations(movie_title, n_recommendations=10)
    # Obtenir le titre et la couverture du film choisi par l'utilisateur
    movie_title = movie_titles[movie_id]
    movie_cover = movie_covers[movie_id]
    # Créer un dictionnaire pour stocker les titres et les couvertures des films recommandés
    result = {
        "user_choice": {"title": movie_title, "cover": movie_cover},
        "recommendations": [{"title": find_key(i, movie_idx), "cover": find_key(i, cover_idx)} for i in similar_movies]

    }
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