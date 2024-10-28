
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import pymongo
from pymongo import MongoClient


# Définir un User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Faire la requête avec le User-Agent
page = requests.get("https://www.imdb.com/chart/boxoffice", headers=headers)
soup = bs(page.content, 'lxml')

# Extraire les liens et les titres
links = [a['href'] for a in soup.find_all('a', class_='ipc-title-link-wrapper')]
titles = [h3.get_text() for h3 in soup.find_all('h3', class_='ipc-title__text')[1:11]]

# Nettoyer les titres pour enlever les numéros
cleaned_titles = [re.sub(r'^\d+\.\s*', '', title) for title in titles]
cleaned_links = [link.split('/')[2].split('?')[0].replace('tt', '') for link in links]

# Récupérations des couvertures, genres et année sur chacune des pages
cover_list = []
genres_list = []
year_list = []

for imbd_ref in cleaned_links:
    page = requests.get(f"http://www.imdb.com/title/tt{imbd_ref}/", headers=headers)
    soup = bs(page.content, 'lxml')

    # Récupération de l'image
    image = soup.find('img', class_="ipc-image")
    link_img = image['src'] if image and 'src' in image.attrs else None

    cover_list.append(link_img)

    # Récupération des genres
    genres = soup.find_all('span', class_='ipc-chip__text')
    movie_genres = [i.text for i in genres[:-1]]
    genres_list.append(movie_genres)

    # Récupération des années
    years = soup.find('a', {
        'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color',
        'tabindex': '0',
        'aria-disabled': 'false',
        'href': f'/title/tt{imbd_ref}/releaseinfo?ref_=tt_ov_rdat'
    })

    year_list.append(years.text if years else None)

# GESTION DE LA MISE A JOUR DE NOTRE BASE DE DONNEES MONGODB
client = MongoClient(
    host="mongodb",
    port=27017,
    username="antoine",
    password="pela"
)

db = client['reco_movies']

# Récupérer le movieId maximum dans la collection movies
max_movie_id_doc = db.movies.find().sort("movieId", pymongo.DESCENDING).limit(1)
max_movie_id = max_movie_id_doc[0]['movieId'] if max_movie_id_doc.count() > 0 else 0

# Insertion des films dans MongoDB
for title, year, genres, cover_link, imdb in zip(cleaned_titles, year_list, genres_list, cover_list, cleaned_links):
    existing_movie = db.movies.find_one({'title': title, 'year': year})

    if existing_movie:
        print(f"Le film {title} - {year} est déjà présent dans la collection movies.")
    else:
        max_movie_id += 1
        db.movies.insert_one({
            'movieId': max_movie_id,
            'title': title,
            'genres': genres,
            'year': year
        })
        db.links.insert_one({
            'movieId': max_movie_id,
            'imdbId': imdb,
            'cover_link': cover_link
        })
        print(f"Insertion du film {title}.")

