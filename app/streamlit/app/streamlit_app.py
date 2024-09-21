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
          username = st.text_input("Nom d'utilisateur")
          password = st.text_input("Mot de passe", type="password")
          submitted = st.form_submit_button("S'inscrire")

          if submitted:
            
               # Convertir le nom d'utilisateur en minuscules avant l'envoi
               normalized_username = username.lower()

               response = requests.post("http://fastapi:8000/auth/", json= {"username":normalized_username, "password": password})
               result = response.json()

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
          username = st.text_input("Nom d'utilisateur")
          password = st.text_input("Mot de passe", type="password")
          submitted = st.form_submit_button("Se connecter")

          if submitted:

               # Convertir le nom d'utilisateur en minuscules avant l'envoi
               normalized_username = username.lower()

               response = requests.post("http://fastapi:8000/auth/token", data= {"username":normalized_username, "password": password})

               if response.status_code == 200:  # Utilisateur coonecté
                    st.success(f"Connexion réussie ! Bienvenue {username}. Vous pouvez maintenant poursuivre sur les prochaines pages.")
                    st.balloons()
                    st.session_state.is_logged_in = True

               elif response.status_code == 401:  # Erreur d'utilisateur déjà enregistré
                    st.error("Utilisateur inconnu. Veuillez vous inscrire.")
               else:
                    st.error("Problème de connexion. Veuillez essayer ultérieurement")

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
               user_id_input = form.text_input("Numéro utilisateur")
               submitted = form.form_submit_button("Envoyer")

          if submitted:

               # Envoyer la requête à l'API
               req = requests.post("http://fastapi:8000/predict/", data={'userId': user_id_input})
               result = req.json()

               # Affichage des résultats sous forme de liste numérotée
               st.write("Voici les recommandations de films :")

               # Nombre total de films à afficher
               num_movies = len(result)

               # Créer des colonnes pour le tableau
               cols = st.columns(5)  # 5 colonnes

               # Afficher les films dans un tableau de 5 colonnes
               for i, (key, value) in enumerate(result.items()):
                    col_index = i % 5  # Calculer l'index de colonne
                    with cols[col_index]:  # Utiliser la colonne correspondante
                         st.image(value["cover"], caption=value["title"], use_column_width=True)

               # Si nous avons atteint la fin d'une ligne (2 lignes ici), nous pouvons faire une pause
               if (i + 1) % 5 == 0 and (i + 1) < num_movies:
                    st.write("")  # Ajouter une ligne vide pour séparer les lignes

     elif page == pages[5]:
          st.header("Page 5")
          # Contenu de la page 5


