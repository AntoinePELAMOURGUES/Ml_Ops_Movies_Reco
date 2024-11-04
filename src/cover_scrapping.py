import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import os

# Définir le chemin vers le sous-dossier et le fichier
data_dir = os.path.join("..", "data", "raw")  # Chemin relatif vers le dossier
links_file = os.path.join(data_dir, "processed_links.csv")

def scrapping_cover(links_file):
    df = pd.read_csv(links_file)
    print("Dataset links chargé")
    imdbId_list = df["imdbId"].tolist()
    cover_list = []
    for imbd_ref in imdbId_list:
        # Définir un User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Faire la requête avec le User-Agent
        page = requests.get(f"http://www.imdb.com/title/tt{imbd_ref}/", headers=headers)

        # Analyser le contenu de la page
        soup = bs(page.content, 'lxml')

        image = soup.find('img', class_="ipc-image")

        if image and 'src' in image.attrs:  # Vérifie si l'image existe et si l'attribut 'src' est présent
            link_img = image['src']
        else:
            link_img = None  #

        cover_list.append(link_img)
        df['cover_link'] = cover_list
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
    scrapping_cover(links_file)