import pandas as pd
import os
import numpy as np


# Chargement des datasets
# Définir le chemin vers le sous-dossier et le fichier
data_dir = os.path.join("data", "raw")  # Chemin relatif vers le dossier
ratings_file = os.path.join(data_dir, "ratings.csv")
movies_file = os.path.join(data_dir, "movies.csv")
links_file = os.path.join(data_dir, "links.csv")



def bayesienne_mean(df, M, C):
    '''
    𝑀  = moyenne brute des notes des films.
    𝐶  = moyenne de la quantité de notes.
    '''
    moy_ba = (C * M + df.sum()) / (C + df.count())
    return moy_ba

def preprocessing_ratings(ratings_file) -> pd.DataFrame:
    """Lecture fichier csv et application de la moyenne Bayesienne."""
    df = pd.read_csv(ratings_file)
    print("Dataset ratings chargé")

    # Quantité de notes par chaque film ainsi que la note moyenne par film
    movies_stats = df.groupby('movieId').agg({'rating': ['count', 'mean']})
    movies_stats.columns = ['count', 'mean']

    # Moyenne de la quantité de notes.
    C = movies_stats['count'].mean()

    # Moyenne brute des notes des films.
    M = movies_stats['mean'].mean()

    # Calculer la moyenne bayésienne par film
    movies_stats['bayesian_mean'] = movies_stats.apply(lambda x: bayesienne_mean(df[df['movieId'] == x.name]['rating'], M, C), axis=1)

    # Ajouter la colonne bayesian_mean au DataFrame original
    df = df.merge(movies_stats[['bayesian_mean']], on='movieId', how='left')

    # Remplacer les évaluations originales par les moyennes bayésiennes
    df['rating'] = df['movieId'].apply(lambda x: movies_stats.loc[x, 'bayesian_mean'])
    print("Application de la moyenne Bayesienne sur la colonne rating effectuée")
    # Définir le chemin vers le sous-dossier 'raw' dans le dossier parent 'data'
    output_dir = os.path.join("..", "data", "raw")  # ".." fait référence au dossier parent
    output_file = os.path.join(output_dir, "processed_ratings.csv")

    # Créer le dossier 'raw' s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Enregistrer le DataFrame en tant que fichier CSV
    try:
        df.to_csv(output_file, index=False)  # Enregistrer sans l'index
        print(f"Fichier enregistré avec succès sous {output_file}.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'enregistrement du fichier : {e}")
    return df

def preprocessing_movies(movies_file):
    '''
    Lecture fichier movies, création d'une colonne year, passage des genres en liste de genres
    '''
    df = pd.read_csv(movies_file)
    print("Dataset movies chargé")
    print("Création d'une colonne year et passage des genres en liste de genres")
    # Split sur les pipes
    df['genres'] = df['genres'].apply(lambda x: x.split("|"))
    # Extraction de l'année et mise à jour du titre
    df['year'] = df['title'].str.extract(r'\((\d{4})\)')[0]
    df['title'] = df['title'].str.replace(r' \(\d{4}\)', '', regex=True)
    df.ffill(inplace= True)
    # Définir le chemin vers le sous-dossier 'raw' dans le dossier parent 'data'
    output_dir = os.path.join("..", "data", "raw")  # ".." fait référence au dossier parent
    output_file = os.path.join(output_dir, "processed_movies.csv")

    # Créer le dossier 'raw' s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Enregistrer le DataFrame en tant que fichier CSV
    try:
        df.to_csv(output_file, index=False)  # Enregistrer sans l'index
        print(f"Fichier enregistré avec succès sous {output_file}.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'enregistrement du fichier : {e}")
    return df

def preprocessing_links(links_file):
    '''
    Chargement du dataset et modification type TmdbId en int'''
    df = pd.read_csv(links_file)
    print("Dataset links chargé")
    print('Modification du type de la colonne tmdbId en int')
    df['tmdbId'] = df.tmdbId.fillna(0)
    df['tmdbId'] = df.tmdbId.astype(int)
    # Définir le chemin vers le sous-dossier 'raw' dans le dossier parent 'data'
    output_dir = os.path.join("..", "data", "raw")  # ".." fait référence au dossier parent
    output_file = os.path.join(output_dir, "processed_links.csv")

    # Créer le dossier 'raw' s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Enregistrer le DataFrame en tant que fichier CSV
    try:
        df.to_csv(output_file, index=False)  # Enregistrer sans l'index
        print(f"Fichier enregistré avec succès sous {output_file}.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'enregistrement du fichier : {e}")
    return df


if __name__ == "__main__":
    ratings_df = preprocessing_ratings(ratings_file)
    movies_df = preprocessing_movies(movies_file)
    links_df = preprocessing_links(links_file)