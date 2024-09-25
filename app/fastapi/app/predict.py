import pandas as pd
import os
import json
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import process
from sklearn.decomposition import TruncatedSVD
from fastapi import Request, APIRouter, HTTPException
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
from pydantic import BaseModel
import re

# ROUTEUR POUR GERER LES ROUTES PREDICT

router = APIRouter(
    prefix='/predict',  # Préfixe pour toutes les routes dans ce routeur
    tags=['predict']    # Tag pour la documentation
)

# ---------------------------------------------------------------

# ENSEMBLE DES FONCTIONS UTILISEES

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

# Chargement du dernier modèle
def load_model(directory = "/app/model") :
    """Charge le modèle avec la dernière version à partir d'un répertoire."""
    # Vérifier si le répertoire existe
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    # Liste des fichiers dans le répertoire
    files = os.listdir(directory)
    # Filtrer les fichiers pour ne garder que ceux qui correspondent au modèle knn
    model_files = [f for f in files if f.startswith("model_knn") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("Aucun modèle KNN trouvé dans le répertoire.")
    # Charger le modèle
    filepath = os.path.join(directory, model_files)
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
        print(f'Modèle chargé depuis {filepath}')
    return model

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

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Predictions si utilisateur connu
def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine', k=10):
    """
    Trouve les k voisins les plus proches pour un ID de film donné.

    Args:
        movie_id: ID du film d'intérêt
        X: matrice d'utilité utilisateur-article (matrice creuse)
        k: nombre de films similaires à récupérer
        metric: métrique de distance pour les calculs kNN

    Output: retourne une liste des k ID de films similaires
    """
    # Transposer la matrice X pour que les films soient en lignes et les utilisateurs en colonnes
    X = X.T
    neighbour_ids = []  # Liste pour stocker les ID des films similaires

    # Obtenir l'index du film à partir du mapper
    movie_ind = movie_mapper[movie_id]

    # Extraire le vecteur correspondant au film spécifié
    movie_vec = X[movie_ind]

    # Vérifier si movie_vec est un tableau NumPy et le remodeler en 2D si nécessaire
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)  # Reshape pour avoir une forme (1, n_features)

    # Initialiser NearestNeighbors avec k+1 car nous voulons inclure le film lui-même dans les voisins
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)

    # Ajuster le modèle sur la matrice transposée (films comme lignes)
    kNN.fit(X)

    # Trouver les k+1 voisins les plus proches (y compris le film d'intérêt)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)

    # Collecter les ID des films parmi les voisins trouvés
    for i in range(0, k):  # Boucler jusqu'à k pour obtenir seulement les films similaires
        n = neighbour.item(i)  # Obtenir l'index du voisin
        neighbour_ids.append(movie_inv_mapper[n])  # Mapper l'index à l'ID du film

    neighbour_ids.pop(0)  # Retirer le premier élément qui est l'ID du film original

    return neighbour_ids  # Retourner la liste des ID de films similaires

# Focntion qui regroupe les recommandations
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Because you watched {title}:")
    print(movies['title'].iloc[similar_movies])

def validate_userId(userId):
    # Vérifier si userId est dans la plage valide
    if userId < 1 or userId > 138493:
        return "Le numéro d'utilisateur doit être compris entre 1 et 138493."
    return None

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

ratings = read_ratings('ratings.csv')
movies = read_movies('movies.csv')
links = read_links('links2.csv')
model_svd = load_model()
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
svd = TruncatedSVD(n_components=20, n_iter=10)
Q = svd.fit_transform(X.T)
Z = Q.T
genres = set(g for G in movies['genres'] for g in G)
for g in genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))
movie_genres = movies.drop(columns=['movieId', 'title','genres'])
movie_idx = dict(zip(movies['title'], list(movies.index)))
cosine_sim = cosine_similarity(movie_genres, movie_genres)
print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")

# ---------------------------------------------------------------

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


