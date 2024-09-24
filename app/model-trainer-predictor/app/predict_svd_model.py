from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
import pandas as pd
from surprise.model_selection import cross_validate
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def read_ratings(ratings_csv: str, data_dir: str = "/app/data/") -> pd.DataFrame:
    """Lit le fichier CSV contenant les évaluations des films."""
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings chargé")
    return data

def read_movies(movies_csv: str, data_dir: str = "/app/data/") -> pd.DataFrame:
    """Lit le fichier CSV contenant les informations sur les films."""
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies chargé")
    return df

def read_links(links_csv: str, data_dir: str = "/app/data/") -> pd.DataFrame:
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

def train_model(df: pd.DataFrame) -> SVD:
    """Entraîne le modèle de recommandation sur les données fournies."""
    # Diviser les données en ensembles d'entraînement et de test
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader=reader)

    # Extraire le Trainset
    trainset = data.build_full_trainset()

    model = SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)

    # Entraîner le modèle
    model.fit(trainset)

    print("Début de la cross-validation")

    # Effectuer la validation croisée sur le Trainset
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)

    # Afficher les résultats
    mean_rmse = cv_results['test_rmse'].mean()
    print("Moyenne des RMSE :", mean_rmse)

    return model, mean_rmse

def save_model(model: SVD, filepath: str, version: str) -> None:
    """Sauvegarde le modèle entraîné dans un fichier."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Modifier le nom du fichier pour inclure la version
    base, ext = os.path.splitext(filepath)
    versioned_filepath = f"{base}_{version}{ext}"

    with open(versioned_filepath, 'wb') as file:
        pickle.dump(model, file)
        print(f'Modèle sauvegardé sous {versioned_filepath}')

def save_best_score(score: float, filepath: str) -> None:
    """Sauvegarde le meilleur score dans un fichier."""
    with open(filepath, 'w') as file:
        file.write(str(score))

def load_best_score(filepath: str) -> float:
    """Charge le meilleur score à partir d'un fichier."""
    if not os.path.exists(filepath):
        return float('inf')  # Valeur par défaut très haute pour que n'importe quel score soit meilleur
    with open(filepath, 'r') as file:
        return float(file.read().strip())

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
    links = read_links('links2.csv')
    df = pd.merge(ratings[['userId', 'movieId', 'rating']], movies[['movieId', 'title']], on='movieId', how='left')
    df = df.merge(links, on = 'movieId', how = 'left')
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['movieId'] = movie_encoder.fit_transform(df['movieId'])
    df.to_csv('/app/data/preprocessing_df.csv')
    # Sauvegarde du LabelEncoder dans un fichier
    with open('/app/model/label_encoder.pkl', 'wb') as file:
        pickle.dump(movie_encoder, file)
    # Entraînement du modèle et récupération de la moyenne des scores
    model, mean_rmse = train_model(df)

    # Chemin du fichier pour sauvegarder le meilleur score et la version
    score_file_path = "best_rmse_score.txt"
    version_file_path = "model_version.txt"

    # Charger le meilleur score actuel
    best_score = load_best_score(score_file_path)

    # Comparer et sauvegarder si le nouveau score est meilleur
    if mean_rmse < best_score:
        print("Nouveau meilleur score trouvé !")
        save_best_score(mean_rmse, score_file_path)

        # Charger la version actuelle et l'incrémenter
        current_version = read_version(version_file_path)
        new_version = current_version + 1

        # Sauvegarder le modèle avec la nouvelle version
        save_model(model, "app/model/model_SVD.pkl", f"v{new_version}")

        # Écrire la nouvelle version dans le fichier
        write_version(version_file_path, new_version)

        print(f"Nouvelle version sauvegardée : v{new_version}")

    else:
        print("Le meilleur score actuel est toujours valide :", best_score)