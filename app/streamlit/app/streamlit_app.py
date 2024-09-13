import streamlit as st
import requests
import os
from streamlit_jwt_authenticator import Authenticator
import streamlit.components.v1 as components

# Setup web page
st.set_page_config(
     page_title="API de Recommndation de films",
    #  page_icon=APP_ICON_URL,
     layout="wide",
)

# Création et mise en forme de notre Sidebar
st.sidebar.title ("Sommaire")

pages = ["Contexte & Objectifs", "Choix du modèle", "Gestion BDD", "API", "Testing & Monitoring"]

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


if page == pages[0]:

     st.write("Page 1")

elif page == pages[1]:

     st.write("Page 2")

elif page == pages[2]:

     st.write("Page 3")

elif page == pages[3]:

     st.title("Bonjour ✌️. On se connait ?")

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
                    elif response.status_code == 400:  # Erreur d'utilisateur déjà enregistré
                         st.error("Nom d'utilisateur déjà enregistré. Veuillez choisir un autre identifiant")
                    else:  # Autres erreurs
                         st.error("Une erreur est survenue. Veuillez réessayer.")
               else:
                    st.warning("Veuillez entrer un nom d'utilisateur.")

     # Concerne la partie identification
     st.markdown("---")

     # Interface de connexion

     def get_access_token(username: str, password: str):
          url = "http://fastapi:8000/token"  # Remplacez par l'URL de votre API FastAPI
          response = requests.post(url, data={"username": username, "password": password})

          if response.status_code == 200:
               return response.json()["access_token"]
          else:
               st.error("Nom d'utilisateur ou mot de passe incorrect.")
               return None

     st.title("Connexion à l'application")

     username = st.text_input("Nom d'utilisateur")
     password = st.text_input("Mot de passe", type="password")

     if st.button("Se connecter"):
          token = get_access_token(username, password)
          if token:
               st.session_state["access_token"] = token
               st.session_state["authentication_status"] = True
               st.success("Connexion réussie !")
          else:
               st.session_state["authentication_status"] = False

     def validate_token(token: str):
          url = "http://fastapi:8000/"
          headers = {"Authorization": f"Bearer {token}"}
          response = requests.get(url, headers=headers)

          if response.status_code == 200:
               return response.json()  # Retourne les données de l'utilisateur
          else:
               st.error("Token invalide ou expiré.")
               return None

     if "access_token" in st.session_state:
          user_info = validate_token(st.session_state["access_token"])

          if user_info:
               st.write("Bienvenue dans notre application de recommandation de films !!")
               st.write("Magnifique, ça marche !")
               # Affichez d'autres contenus ici
          else:
               st.session_state["authentication_status"] = False
     else:
          st.warning("Veuillez vous connecter pour accéder à l'application.")

     if st.button("Se déconnecter"):
          st.session_state.clear()  # Efface toutes les données de session
          st.success("Vous êtes déconnecté.")

    # langue = st.selectbox("Choose your language",
    # ("English", "Français"))

    # logo = "logo_rakuten.png"

    # # # Affichage de l'image en haut de la page
    # st.image(logo)

    # if langue == "English":

    #     option = st.selectbox("Choose the model",
    #     ("Text : SGDClassifier", "Image : EfficientNetB1", "Text & Image : Bert & EfficientNetV2L"))

    #     st.title("Welcome to the Rakuten website")

    #     st.header("What do you want to sell today ?")

    #     st.write("A precise title and the right category are the best ways for your future buyers to see your ad!")

    #     # Create a form
    #     form = st.form("product_info")

    #     with form:
    #         designation = form.text_input("Désignation *")
    #         description = form.text_input("Description")
    #         upload = form.file_uploader("Upload your object image",
    #                                     type=['png', 'jpeg', 'jpg'], accept_multiple_files=False)

    #         submitted = form.form_submit_button("Send")
    # else:

    #     option = st.selectbox("Choisissez le type de modèle",
    #     ("Texte : SGDClassifier", "Image : EfficientNetB1", "Texte & Image : Bert & EfficientNetV2L"))

    #     st.title("Bienvenue sur le site Rakuten")

    #     st.header("Que souhaitez-vous vendre aujourd'hui ?")

    #     st.write("Un titre précis et la bonne catégorie, c'est le meilleur moyen pour que vos futurs acheteurs voient votre annonce !")

    #     # Create a form
    #     form = st.form("product_info")

    #     with form:
    #         designation = form.text_input("Titre *")
    #         description = form.text_input("Description")
    #         upload = form.file_uploader("Chargez l'image de votre objet",
    #                                     type=['png', 'jpeg', 'jpg'], accept_multiple_files=False)

    #         submitted = form.form_submit_button("Envoyer")

    # if submitted:
    #     concat_text = designation + ' ' + description
    #     if langue == 'Français':
    #         langue = 'french'
    #     else:
    #         langue = 'english'

    #     if upload:
    #         files = {"file": upload.getvalue()}
    #         if option == 'Texte & Image : Bert & EfficientNetV2L' or option == "Text & Image : Bert & EfficientNetV2L":
    #             req = requests.post("http://backend:8000/predict_multimodal", data={'texte' : concat_text,'langue': langue} , files=files)  # A UTILISER SI DOCKER
    #             # req = requests.post("http://localhost:8000/predict_multimodal", data={'texte' : concat_text,'langue': langue} , files=files)

    #         elif option ==  "Image : EfficientNetB1":
    #             req = requests.post("http://backend:8000/predict_image", data={'texte' : concat_text,'langue': langue} , files=files)
    #             # req = requests.post("http://localhost:8000/predict_image", data={'texte' : concat_text,'langue': langue} , files=files)


            # else:
            #     req = requests.post("http://backend:8000/predict_texte", data={'texte' : concat_text,'langue': langue} , files=files)
            #     # req = requests.post("http://localhost:8000/predict_texte", data={'texte' : concat_text,'langue': langue} , files=files)


            # resultat = req.json()

            # top1_label = resultat["meilleur_Label_1"]
            # top1_txt = resultat["meilleur_label_texte_1"]
            # top1_score = round(resultat["meilleur_Score_1"])
            # top2_label = resultat["meilleur_Label_2"]
            # top2_txt = resultat["meilleur_label_texte_2"]
            # top2_score = round(resultat["meilleur_Score_2"])
            # top3_label = resultat["meilleur_Label_3"]
            # top3_txt = resultat["meilleur_label_texte_3"]
            # top3_score = round(resultat["meilleur_Score_3"])

            # # Display the image and prediction results
            # col1, col2 = st.columns(2)
            # col1.image(Image.open(upload))
            # with col2:
            #     st.radio("__Choisissez une catégorie suggérée__\n",
            #                 [f"{top1_txt} ({top1_label})", f"{top2_txt} ({top2_label})", f"{top3_txt} ({top3_label})"], captions = [f"Probabilité : {top1_score}%", f"Probabilité : {top2_score}%", f"Probabilité : {top3_score}%"])
