import streamlit as st
import requests
import os
import streamlit.components.v1 as components

# Setup web page
st.set_page_config(
     page_title="API de Recommandation de films",
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
          username = st.text_input("Nom d'utilisateur")
          password = st.text_input("Mot de passe", type="password")
          submitted = st.form_submit_button("S'inscrire")

          if submitted:
               if username and password:
                    response = requests.post("http://fastapi:8000/auth/", json= {"username":username, "password": password})
                    result = response.json()

                    if response.status_code == 201:  # Utilisateur créé avec succès
                         st.success(f"Inscription réussie ! Bienvenue {username}. Vous pouvez maintenant vous connecter.")
                         st.balloons()
                    elif response.status_code == 400:  # Erreur d'utilisateur déjà enregistré
                         st.error("Nom d'utilisateur déjà enregistré. Veuillez choisir un autre identifiant")
                    else:  # Autres erreurs
                         st.error("Une erreur est survenue. Veuillez réessayer.")
               else:
                    st.warning("Veuillez entrer un nom d'utilisateur.")

     # Concerne la partie identification
     st.markdown("---")

     # Formulaire de connexion
     with st.form("connexion_form"):
          st.header("Connexion")
          username = st.text_input("Nom d'utilisateur")
          password = st.text_input("Mot de passe", type="password")
          submitted = st.form_submit_button("Se connecter")

          if submitted:
               if username and password:
                    response = requests.post("http://fastapi:8000/auth/token", data= {"username":username, "password": password})

                    if response.status_code == 200:  # Utilisateur coonecté
                         st.success(f"Connexion réussie ! Bienvenue {username}. Vous pouvez maintenant poursuivre sur les prochaines pages.")
                         st.balloons()
                         st.session_state.is_logged_in = True

                    elif response.status_code == 401:  # Erreur d'utilisateur déjà enregistré
                         st.error("Utilisateur inconnu. Veuillez vous inscrire.")
                    else:
                         st.error("Problème de connexion. Veuillez essayer ultérireurement")

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
               # Vérification que user_id est un entier et dans la plage spécifiée
               try:
                    user_id = int(user_id_input)  # Convertir en entier
                    if user_id < 1 or user_id > 138493:
                         st.error("Le numéro d'utilisateur doit être compris entre 1 et 138493.")
                    else:
                         # Si la validation passe, envoyer la requête à l'API
                         req = requests.post("http://fastapi:8000/predict/", data={'userId': user_id})
                         result = req.json()

                         # Affichage des résultats sous forme de liste numérotée
                         st.write("Voici les recommandations de films :")

                         for index, title in result.items():
                              st.markdown(f"{index}. {title}")
               except ValueError:
                    st.error("Veuillez entrer un numéro d'utilisateur valide (un entier).")

     elif page == pages[5]:
          st.header("Page 5")
          # Contenu de la page 5


