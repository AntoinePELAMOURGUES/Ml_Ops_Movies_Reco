import pandas as pd
import os
import time
import pickle
from surprise import Reader, Dataset, accuracy, SVD
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from surprise.model_selection import cross_validate

def read_ratings(ratings_csv: str, data_dir: str = "/home/antoine/Ml_Ops_Movies_Reco/src/data/data/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les évaluations des films.

    :param ratings_csv: Nom du fichier CSV contenant les évaluations.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les évaluations.
    """
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings chargé")
    return data

def read_movies(movies_csv: str, data_dir: str = "/home/antoine/Ml_Ops_Movies_Reco/src/data/data/raw") -> pd.DataFrame:
    """
    Lit le fichier CSV contenant les informations sur les films.

    :param movies_csv: Nom du fichier CSV contenant les informations sur les films.
    :param data_dir: Répertoire où se trouve le fichier CSV.
    :return: DataFrame contenant les informations sur les films.
    """
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies chargé")
    return df

def create_user_matrix(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice utilisateur à partir des évaluations et des informations sur les films.

    :param ratings: DataFrame contenant les évaluations.
    :param movies: DataFrame contenant les informations sur les films.
    :return: DataFrame fusionné et prétraité.
    """
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    mlb = MultiLabelBinarizer()

    # Fusionner les évaluations et les informations sur les films
    df = ratings.merge(movies[['movieId', 'genres']], on="movieId", how="left")
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['movieId'] = movie_encoder.fit_transform(df['movieId'])

    # Traitement des genres
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres').str.split('|')),
                               columns=mlb.classes_, index=df.index))
    df = df.drop("(no genres listed)", axis=1, errors='ignore')  # Ignore si la colonne n'existe pas

    print("Dataset fusionnés et prétraités")
    return df

def cross_validation_model(df: pd.DataFrame, model = SVD()):
    reader = Reader(rating_scale = (0.5, 5))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    algo = model
    print("Début de la cross_validation")
    print(cross_validate(algo, data, cv=2))
    return algo

def train_model(df: pd.DataFrame):
    """
    Entraîne le modèle de recommandation sur les données fournies.

    :param df: DataFrame contenant les données d'entraînement.
    :param model: Modèle de recommandation à entraîner (par défaut SVD).
    :return: Modèle entraîné.
    """
    start = time.time()

    # Préparer le lecteur
    reader = Reader(rating_scale=(1, 5))

    # Charger les données depuis le DataFrame
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    # Construire l'ensemble d'entraînement
    trainset = data.build_full_trainset()
    # Entraîner le modèle
    print("Début entrainement du modèle")
    algo = SVD()
    algo.fit(trainset)

    # # Évaluer le modèle
    # print("Début évaluation du modèle")
    # # Construire l'ensemble de test anti-test
    # testset = trainset.build_anti_testset()
    # # Faire des prédictions sur l'ensemble de test
    # predictions = model.test(testset)

    # # Calculer RMSE et MAE
    # rmse = accuracy.rmse(predictions)
    # mae = accuracy.mae(predictions)

    # print(f"Score RMSE : {rmse}")
    # print(f"Score MAE : {mae}")

    end = time.time()
    print(f"Temps d'exécution: {end - start} secondes")

    return algo

def save_model(model, filepath: str) -> None:
    """
    Sauvegarde le modèle entraîné dans un fichier.

    :param model: Modèle à sauvegarder.
    :param filepath: Chemin complet du fichier où le modèle sera sauvegardé.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Créer le dossier si nécessaire
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
        print(f'Modèle sauvegardé sous {filepath}')

if __name__ == "__main__":
    # Chargement des données
    ratings = read_ratings('ratings.csv')
    movies = read_movies('movies.csv')
    # Création de la matrice utilisateur
    df = create_user_matrix(ratings, movies)
    # # Cross_validation
    model = cross_validation_model(df, model = SVD())
    # # Entrainenement du modèle
    model = train_model(df)
    # Sauvegarde du modèle
    save_model(model, '/home/antoine/Ml_Ops_Movies_Reco/models/model_SVD.pkl')