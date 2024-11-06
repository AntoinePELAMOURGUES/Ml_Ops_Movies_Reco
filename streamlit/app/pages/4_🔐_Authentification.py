import streamlit as st
import requests

# Initialisation de l'état de connexion si ce n'est pas déjà fait
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

st.title("Authentification")

st.header("Bonjour✌️. On se connait?")

st.warning("Veuillez vous inscrire ou vous connecter.")
# Formulaire d'inscription
with st.form("registration_form", clear_on_submit = True ):
    st.header("Inscription")
    st.write("Règles de sécurité:")
    st.markdown("""
                - Le nom d'utilisateur ne doit contenir que des lettres, chiffres et underscores.
                - Le mot de passe doit contenir au moins 12 caractères, un chiffre, une majuscule et un cartère spécial.
                """)
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    submitted = st.form_submit_button("S'inscrire", use_container_width= True)

    if submitted:

        # Convertir le nom d'utilisateur en minuscules avant l'envoi
        normalized_username = username.lower()

        response = requests.post("http://fastapi:6060/auth/", json= {"username":normalized_username, "password": password})
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
with st.form("connexion_form", clear_on_submit=True):
    st.header("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    submitted = st.form_submit_button("Se connecter", use_container_width = True)

    if submitted:

        # Convertir le nom d'utilisateur en minuscules avant l'envoi
        normalized_username = username.lower()

        response = requests.post("http://fastapi:6060/auth/token", data= {"username":normalized_username, "password": password})
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