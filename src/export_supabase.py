import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialiser le client Supabase
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def upload_csv_to_supabase(csv_file_path: str):
    """Charge un fichier CSV et insère ses données dans les tables appropriées de Supabase."""
    # Lire le fichier CSV dans un DataFrame Pandas
    df = pd.read_csv(csv_file_path)

    # Vérifier que le DataFrame n'est pas vide
    if df.empty:
        print("Le fichier CSV est vide.")
        return

    # Insérer les données dans la table ratings
    if 'rating' in csv_file_path:
        # Convertir la colonne timestamp de Unix timestamp à datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        for index, row in df.iterrows():
            data = {
                "userId": int(row["userId"]),
                "movieId": int(row["movieId"]),
                "rating": float(row["rating"]),
                "timestamp": row["timestamp"].isoformat()
            }

            # Insérer la ligne dans Supabase
            response = supabase.from_("ratings").insert(data).execute()



    elif 'movies' in csv_file_path:
        for index, row in df.iterrows():
            data = {
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "year": int(row["year"])
            }

            # Insérer la ligne dans Supabase
            response = supabase.from_("movies").insert(data).execute()


    elif 'links' in csv_file_path:
        for index, row in df.iterrows():
            data = {
                "movieId": int(row["movieId"]),
                "imdbId": int(row["imdbId"]),
                "tmdbId": int(row["tmdbId"]),
                "cover_link": row["cover_link"]
            }

            # Insérer la ligne dans Supabase
            response = supabase.from_("links").insert(data).execute()


# Exemple d'utilisation
if __name__ == "__main__":
    csv_files = [
        os.path.join("..", "data", "raw", "processed_ratings.csv"),
        os.path.join("..", "data", "raw", "processed_movies.csv"),
        os.path.join("..", "data", "raw", "processed_links.csv")
    ]

    for csv_file_path in csv_files:
        upload_csv_to_supabase(csv_file_path)