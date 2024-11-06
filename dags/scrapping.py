from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import os
from supabase import create_client
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def connect_to_supabase():
    """Établit une connexion à Supabase en utilisant les variables d'environnement."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not all([supabase_url, supabase_key]):
        raise ValueError(
            "Les variables d'environnement SUPABASE_URL et SUPABASE_KEY doivent être définies."
        )
    return create_client(supabase_url, supabase_key)

def scrape_imdb():
    """Scrape les données des films depuis IMDb et les insère dans Supabase."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Récupérer la page des box-offices d'IMDb
    page = requests.get("https://www.imdb.com/chart/boxoffice", headers=headers)
    soup = bs(page.content, 'lxml')

    # Extraire les liens et titres des films
    links = [a['href'] for a in soup.find_all('a', class_='ipc-title-link-wrapper')]
    titles = [h3.get_text() for h3 in soup.find_all('h3', class_='ipc-title__text')[1:11]]

    # Nettoyer les titres et extraire les IDs IMDb
    cleaned_titles = [re.sub(r'^\d+\.\s*', '', title) for title in titles]
    cleaned_links = [link.split('/')[2].split('?')[0].replace('tt', '') for link in links]

    cover_list, genres_list, year_list = [], [], []

    # Boucle pour récupérer les détails de chaque film
    for imdb_ref in cleaned_links:
        movie_page = requests.get(f"http://www.imdb.com/title/tt{imdb_ref}/", headers=headers)
        soup_movie = bs(movie_page.content, 'lxml')

        # Récupérer l'image de couverture
        image = soup_movie.find('img', class_="ipc-image")
        link_img = image['src'] if image and 'src' in image.attrs else None
        cover_list.append(link_img)

        # Récupérer les genres du film
        genres = soup_movie.find_all('span', class_='ipc-chip__text')
        movie_genres = [i.text for i in genres[:-1]]
        genres_list.append(movie_genres)

        # Récupérer l'année de sortie du film
        year_elem = soup_movie.find('a', {
            'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color',
            'tabindex': '0',
            'aria-disabled': 'false',
            'href': f'/title/tt{imdb_ref}/releaseinfo?ref_=tt_ov_rdat'
        })
        year_list.append(year_elem.text if year_elem else None)

    # Connexion à Supabase pour insérer les données
    supabase = connect_to_supabase()

    # Récupérer le dernier movieId dans la base de données pour éviter les doublons
    response = supabase.from_("movies").select("movieId").order("movieId", desc=True).limit(1).execute()

    max_movie_id = response.data[0]['movieId'] + 1 if response.data else 1

    # Insertion des films dans Supabase
    for title, year, genres, cover_link, imdb in zip(cleaned_titles, year_list, genres_list, cover_list, cleaned_links):
        existing_movie_response = supabase.from_("movies").select("*").eq("title", title).eq("year", year).execute()

        if existing_movie_response.data:
            print(f"Le film {title} - {year} est déjà présent dans la collection movies.")
        else:
            # Insérer le nouveau film dans la table movies
            supabase.from_("movies").insert({
                'movieId': max_movie_id,
                'title': title,
                'genres': genres,
                'year': year
            }).execute()

            # Insérer le nouveau lien dans la table links
            supabase.from_("links").insert({
                'movieId': max_movie_id,
                'imdbId': imdb,
                'tmdbId': 0,
                'cover_link': cover_link
            }).execute()

            print(f"Insertion du film {title}.")
            max_movie_id += 1

def export_table_to_csv(table_name, csv_file_path):
    """Exporte une table de Supabase vers un fichier CSV."""
    supabase = connect_to_supabase()

    # Récupérer les données de la table spécifiée
    response = supabase.from_(table_name).select("*").execute()

    if response.error:
        print("Erreur lors de la récupération des données:", response.error)
        return

    # Convertir les données en DataFrame Pandas et exporter en CSV
    data = response.data
    df = pd.DataFrame(data)

    df.to_csv(csv_file_path, index=False)
    print(f"Données exportées vers {csv_file_path} avec succès.")

# Arguments par défaut pour le DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 31),
}

# Création du DAG
dag = DAG(
    dag_id='imdb_scraper_dag',
    description='Scraping IMDb and updating datasets',
    tags=['reco_movies'],
    default_args=default_args,
    schedule_interval='@daily',
)

# Tâche pour scraper IMDb
scrape_task = PythonOperator(
    task_id='scrape_imdb_task',
    python_callable=scrape_imdb,
    dag=dag,
)

# Tâche pour exporter la table movies vers CSV
update_movies_task = PythonOperator(
    task_id='update_movies_task',
    python_callable=export_table_to_csv,
    dag=dag,
    op_kwargs={"table_name": "movies", "csv_file_path": "/opt/airflow/data/raw/processed_movies.csv"}
)

# Tâche pour exporter la table links vers CSV
update_links_task = PythonOperator(
    task_id='update_links_task',
    python_callable=export_table_to_csv,
    dag=dag,
    op_kwargs={"table_name": "links", "csv_file_path": "/opt/airflow/data/raw/processed_links.csv"}
)

# Définir l'ordre d'exécution des tâches dans le DAG
scrape_task >> update_movies_task >> update_links_task
