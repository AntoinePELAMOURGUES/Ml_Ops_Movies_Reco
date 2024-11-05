import os
import sys
import pandas as pd
import pytest
from unittest import mock
# Ajouter le chemin du dossier src au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing  import preprocessing_ratings, preprocessing_movies, preprocessing_links

# Chemin du répertoire de test
TEST_DATA_DIR = "./tests/data/raw"
TEST_RATINGS_FILE = os.path.join(TEST_DATA_DIR, "ratings.csv")
TEST_MOVIES_FILE = os.path.join(TEST_DATA_DIR, "movies.csv")
TEST_LINKS_FILE = os.path.join(TEST_DATA_DIR, "links.csv")

@pytest.fixture(scope='module', autouse=True)
def setup_test_data():
    # Créer le répertoire de test et les fichiers CSV nécessaires
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Créer un fichier ratings.csv pour les tests
    ratings_data = {
        'userId': [1, 1, 2, 2],
        'movieId': [1, 2, 1, 3],
        'rating': [4.0, 5.0, 4.0, 3.0]
    }
    pd.DataFrame(ratings_data).to_csv(TEST_RATINGS_FILE, index=False)

    # Créer un fichier movies.csv pour les tests
    movies_data = {
        'movieId': [1, 2, 3],
        'title': ['Movie A (2000)', 'Movie B (2001)', 'Movie C (2002)'],
        'genres': ['Drama|Action', 'Comedy|Romance', 'Thriller']
    }
    pd.DataFrame(movies_data).to_csv(TEST_MOVIES_FILE, index=False)

    # Créer un fichier links.csv pour les tests
    links_data = {
        'movieId': [1, 2, 3],
        'tmdbId': [1001, None, 1003]
    }
    pd.DataFrame(links_data).to_csv(TEST_LINKS_FILE, index=False)

    yield

    # Nettoyage après les tests
    for file in [TEST_RATINGS_FILE, TEST_MOVIES_FILE, TEST_LINKS_FILE]:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(TEST_DATA_DIR)

def test_preprocessing_ratings():
    df = preprocessing_ratings(TEST_RATINGS_FILE)

    assert df is not None
    assert 'bayesian_mean' in df.columns
    assert df['rating'].notnull().all(), "Les évaluations ne doivent pas être nulles."

def test_preprocessing_movies():
    df = preprocessing_movies(TEST_MOVIES_FILE)

    assert df is not None
    assert 'year' in df.columns
    assert isinstance(df['genres'].iloc[0], list), "Les genres doivent être une liste."

def test_preprocessing_links():
    df = preprocessing_links(TEST_LINKS_FILE)

    assert df is not None
    assert df['tmdbId'].dtype == int, "La colonne tmdbId doit être de type int."