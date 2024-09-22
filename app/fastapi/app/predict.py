import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from fastapi import Request, APIRouter, HTTPException
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, CollectorRegistry
import time

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

def read_links(links_csv: str, data_dir: str = "/code/app/data/") -> pd.DataFrame:
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

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
mlb = MultiLabelBinarizer()

def create_user_matrix(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice utilisateur à partir des évaluations et des informations sur les films.

    :param ratings: DataFrame contenant les évaluations.
    :param movies: DataFrame contenant les informations sur les films.
    :param links : DataFrame contenant le lien des affiches
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
links = read_links('links2.csv')

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

def validate_userId(userId):
    # Vérifier si userId est un entier
    if not isinstance(userId, int):
        return "Le numéro d'utilisateur doit être un entier."

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


# Route pour récupérer les tops 10 de notre modèle
@router.post("/")
async def predict(request: Request) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.

    :param request: La requête HTTP contenant l'ID utilisateur.
    :return: Un dictionnaire avec les recommandations de films.
    """

    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/predict').inc()

    # Récupération des données Streamlit
    try:
        # Récupération des données du formulaire
        request_data = await request.form()
        print({'request_data' : request_data})

        user_id = request_data.get('userId')
        # Récupérer et convertir en entier
        # valider l'user_id
        userId_error = validate_userId(user_id)
        if userId_error:
            error_counter.labels(error_type='invalid_userId').inc()
            raise HTTPException(status_code=400, detail=userId_error)
        recommendations = get_top_n_recommendations(user_id)
        top_n_movies_titles = movies[movies['movieId'].isin(recommendations)]['title'].tolist()
        top_n_movies_covers = links[links['movieId'].isin(recommendations)]['cover_link'].tolist()
         # Créer un dictionnaire associant chaque titre à son lien de couverture
        result = {
            str(i): {
                "title": title,
                "cover": cover
            } for i, (title, cover) in enumerate(zip(top_n_movies_titles, top_n_movies_covers), 1)
        }
        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(result))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques
        status_code_counter.labels(status_code = "200").inc()
        duration_of_requests_histogram.labels(method='POST', endpoint='/predict', user_id=str(user_id)).observe(duration)
        response_size_histogram.labels(method='POST', endpoint='/predict').observe(response_size)

        return result

    except ValueError:
        error_counter.labels(error_type='invalid_user_id').inc()
        status_code_counter.labels(status_code = "400").inc()
        raise HTTPException(status_code=400, detail="Invalid user ID")
    except KeyError:
        error_counter.labels(error_type='user_id_not_found').inc()
        status_code_counter.labels(status_code = "400").inc()
        raise HTTPException(status_code=400, detail="User ID not found in the request")

