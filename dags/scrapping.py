from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import pymongo
import os

# MongoDB client setup
client = pymongo.MongoClient(
    host="mongodb",
    port=27017,
    username="antoine",
    password="pela"
)
db = client['reco_movies']

def scrape_imdb():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    page = requests.get("https://www.imdb.com/chart/boxoffice", headers=headers)
    soup = bs(page.content, 'lxml')

    links = [a['href'] for a in soup.find_all('a', class_='ipc-title-link-wrapper')]
    titles = [h3.get_text() for h3 in soup.find_all('h3', class_='ipc-title__text')[1:11]]

    cleaned_titles = [re.sub(r'^\d+\.\s*', '', title) for title in titles]
    cleaned_links = [link.split('/')[2].split('?')[0].replace('tt', '') for link in links]

    cover_list, genres_list, year_list = [], [], []

    for imdb_ref in cleaned_links:
        movie_page = requests.get(f"http://www.imdb.com/title/tt{imdb_ref}/", headers=headers)
        soup_movie = bs(movie_page.content, 'lxml')

        image = soup_movie.find('img', class_="ipc-image")
        link_img = image['src'] if image and 'src' in image.attrs else None

        cover_list.append(link_img)

        genres = soup_movie.find_all('span', class_='ipc-chip__text')
        movie_genres = [i.text for i in genres[:-1]]
        genres_list.append(movie_genres)

        year_elem = soup_movie.find('a', {
            'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color',
            'tabindex': '0',
            'aria-disabled': 'false',
            'href': f'/title/tt{imdb_ref}/releaseinfo?ref_=tt_ov_rdat'
        })
        year_list.append(year_elem.text if year_elem else None)

        max_movie_id = db.movies.find().sort("movieId", pymongo.DESCENDING).limit(1)
        # Extraire le movieId
        for doc in max_movie_id:
            max_id = doc['movieId']

        for title, year, genres, cover_link, imdb in zip(cleaned_titles, year_list, genres_list, cover_list, cleaned_links):
            existing_movie = db.movies.find_one({'title': title, 'year': year})

            if existing_movie:
                print(f"Le film {title} - {year} est déjà présent dans la collection movies.")
            else:
                db.movies.insert_one({
                    'movieId': max_id,
                    'title': title,
                    'genres': genres,
                    'year': year
                })
                db.links.insert_one({
                    'movieId': max_id,
                    'imdbId': imdb,
                    'cover_link': cover_link
                })
                print(f"Insertion du film {title}.")
                max_id += 1

def update_datasets():
    movies = db.movies
    links = db.links
    movies_docs = list(movies.find())
    links_docs = list(links.find())

    df_movies = pd.DataFrame(movies_docs)
    df_links = pd.DataFrame(links_docs)

    df_movies.drop(columns=['_id'], inplace=True)
    df_links.drop(columns=['_id'], inplace=True)

    # Exportation du DataFrame au format CSV
    output_dir = "/opt/airflow/data/raw"
    os.makedirs(output_dir, exist_ok=True)  # Assurez-vous que le dossier existe

    csv_file_path_movies = os.path.join(output_dir, "movies.csv")
    csv_file_path_links = os.path.join(output_dir, "links2.csv")

    df_movies.to_csv(csv_file_path_movies, index=False)
    df_links.to_csv(csv_file_path_links, index=False)

    print(f"Exportations réussies : {csv_file_path_movies}, {csv_file_path_links}")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 31),
}

dag = DAG(
    dag_id='imdb_scraper_dag',
    description='Scraping IMDb and updating datasets',
    tags=['reco_movies'],
    default_args=default_args,
    schedule_interval='@daily',
)

scrape_task = PythonOperator(
    task_id='scrape_imdb_task',
    python_callable=scrape_imdb,
    dag=dag,
)

update_datasets_task = PythonOperator(
    task_id='update_csv_files_task',
    python_callable=update_datasets,
    dag=dag,
)

scrape_task >> update_datasets_task