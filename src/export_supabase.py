import os
import pandas as pd
from supabase import create_client, Client

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Fonction pour charger un fichier CSV et insérer les données dans Supabase
def upload_csv_to_supabase(csv_file_path: str):
    """Charge un fichier CSV et insère ses données dans la table ratings de Supabase."""
    # Lire le fichier CSV dans un DataFrame Pandas
    df = pd.read_csv(csv_file_path)

    # Vérifier que le DataFrame n'est pas vide
    if df.empty:
        print("Le fichier CSV est vide.")
        return

    # Insérer les données dans la table ratings
    for index, row in df.iterrows():
        data = {
            "userId": row["userId"],
            "movieId": row["movieId"],
            "rating": row["rating"],
            "timestamp": row["timestamp"]
        }

        # Insérer la ligne dans Supabase
        response = supabase.from_("ratings").insert(data).execute()

        if response.error:
            print(f"Erreur lors de l'insertion de la ligne {index}: {response.error}")
        else:
            print(f"Ligne {index} insérée avec succès.")

# Exemple d'utilisation
if __name__ == "__main__":
    csv_file_path = "/path/to/your/ratings.csv"  # Remplacez par le chemin vers votre fichier CSV
    upload_csv_to_supabase(csv_file_path)