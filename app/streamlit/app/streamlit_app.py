import streamlit as st
import requests


# Setup web page
st.set_page_config(

     page_title="API de Recommndation de films",
     #  page_icon=APP_ICON_URL,

     layout="wide",
)

# Création et mise en forme de notre Sidebar
st.sidebar.title ("Sommaire")

pages = ["Contexte & Objectifs", "Choix du modèle", "Gestion BDD", "Authentification", "API", "Testing & Monitoring"]

page = st.sidebar.radio(":red[RECOMMANDATION DE FILMS :]", pages)

st.sidebar.write(":red[COHORTE :]")
st.sidebar.markdown("""
<div style='line-height: 1.5;'>
Antoine PELAMOURGUES<br>
Kévin HUYNH<br>
Mikhael BENILOUZ<br>
Sarah HEMMEL<br>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.write(":red[Mentor :]")
st.sidebar.write("Maria")

# Variable d'état pour suivre si l'utilisateur est connecté
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False


if page == pages[0]:

     st.write("Page 1")

elif page == pages[1]:

     st.write("Page 2")

elif page == pages[2]:

     st.write("Page 3")

elif page == pages[3]:

     st.header("Bonjour✌️. On se connait?")

     st.warning("Veuillez vous inscrire ou vous connecter.")
     # Formulaire d'inscription
     with st.form("registration_form"):
          st.header("Inscription")
          st.write("Règles de sécurité:")
          st.markdown("""
                      - Le nom d'utilisateur ne doit contenir que des lettres, chiffres et underscores.
                      - Le mot de passe doit contenir au moins 12 caractères, un chiffre, une majuscule et un cartère spécial.
                      """)
          # Utiliser session_state pour stocker les valeurs permettant remise à blanc du formulaire après submitted
          if 'username_reg' not in st.session_state:
               st.session_state.username_reg = ""
          if 'password_reg' not in st.session_state:
               st.session_state.password_reg = ""

          username = st.text_input("Nom d'utilisateur", value=st.session_state.username_reg)
          password = st.text_input("Mot de passe", type="password", value=st.session_state.password_reg)
          submitted = st.form_submit_button("S'inscrire")

          if submitted:

               # Convertir le nom d'utilisateur en minuscules avant l'envoi
               normalized_username = username.lower()

               response = requests.post("http://fastapi:8000/auth/", json= {"username":normalized_username, "password": password})
               result = response.json()
               st.session_state.username_reg = ""
               st.session_state.password_reg = ""
               if response.status_code == 201:  # Utilisateur créé avec succès
                    st.success(f"Inscription réussie ! Bienvenue {username}. Vous pouvez maintenant vous connecter.")
                    st.balloons()
               elif response.status_code == 400:  # Erreur d'utilisateur déjà enregistré ou autres erreurs de validation
                    error_message = response.json().get("detail", "Une erreur est survenue.")
                    st.error(error_message)  # Afficher le message d'erreur détaillé
               else:  # Autres erreurs
                    st.error("Une erreur est survenue. Veuillez réessayer.")

     # Concerne la partie identification
     st.markdown("---")

     # Formulaire de connexion
     with st.form("connexion_form"):
          st.header("Connexion")
          if 'username_conn' not in st.session_state:
            st.session_state.username_conn = ""
          if 'password_conn' not in st.session_state:
               st.session_state.password_conn = ""

          username = st.text_input("Nom d'utilisateur", value=st.session_state.username_conn)
          password = st.text_input("Mot de passe", type="password", value=st.session_state.password_conn)
          submitted = st.form_submit_button("Se connecter")

          if submitted:

               # Convertir le nom d'utilisateur en minuscules avant l'envoi
               normalized_username = username.lower()

               response = requests.post("http://fastapi:8000/auth/token", data= {"username":normalized_username, "password": password})
               st.session_state.username_conn = ""
               st.session_state.password_conn = ""
               if response.status_code == 200:  # Utilisateur coonecté
                    st.success(f"Connexion réussie ! Bienvenue {username}. Vous pouvez maintenant poursuivre sur les prochaines pages.")
                    st.balloons()
                    st.session_state.is_logged_in = True

               else:
                    # Erreur d'utilisateur déjà enregistré
                    error_message = response.json().get("detail", "Une erreur est survenue.")
                    st.error(error_message)  # Afficher le message d'erreur détaillé


# Vérifiez si l'utilisateur est connecté avant d'afficher les pages 4 et 5
if st.session_state.is_logged_in:
     # Afficher le contenu des pages 4 et 5 ici
     if page == pages[4]:

          st.header("Bienvenue sur notre site de recommandation de films")

          logo = "netflix-catalogue.jpg"

          # Affichage de l'image en haut de la page
          st.image(logo)

          st.write("Choisissez un numéro d'utilisateur compris entre 1 et 138493")

          # Create a form
          form = st.form("Infos_utilisateur")

          with form:
               user_id_input = form.number_input("Numéro utilisateur", min_value=1, step=1, format="%d")
               submitted = form.form_submit_button("Envoyer")

          if submitted:

               # Envoyer la requête à l'API
               req = requests.post("http://fastapi:8000/predict/", json={'userId': user_id_input})
               result = req.json()
               # Affichage des meilleurs films de l'utilisateur
               st.write("Mes 5 films préférés:")
               if "best_user_movies" in result:
                    best_movies = result["best_user_movies"]
                    print(best_movies)

                    # Créer des colonnes pour afficher les meilleurs films
                    cols_best_movies = st.columns(5)  # 5 colonnes pour les meilleurs films

                    for i, movie in enumerate(best_movies):
                         col_index = i % 5  # Calculer l'index de colonne
                         with cols_best_movies[col_index]:  # Utiliser la colonne correspondante
                              # Combiner le titre et les genres pour la légende
                              caption = f"{movie['title']} - Genres: {movie['genres']}"
                              st.image(movie["cover"], caption=caption, use_column_width=True)

               # Affichage des recommandations de films
               st.write("Voici les recommandations de films :")
               if "recommendations" in result:
                    recommended_movies = result["recommendations"]

                    # Créer des colonnes pour afficher les recommandations
                    cols_recommended_movies = st.columns(5)  # 5 colonnes pour les recommandations

                    for i, movie in enumerate(recommended_movies):
                         col_index = i % 5  # Calculer l'index de colonne
                         with cols_recommended_movies[col_index]:
                              caption = f"{movie['title']} - Genres: {movie['genres']}"
                              st.image(movie["cover"], caption=movie["title"], use_column_width=True)

     elif page == pages[5]:
          st.header("Page 5")
          # Contenu de la page 5


