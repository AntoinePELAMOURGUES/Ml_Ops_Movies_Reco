import streamlit as st
import requests

if not st.session_state['is_logged_in']:
    st.warning("You need to be logged in to access this page.")
    st.stop()

st.write("Parfait 💪. Vos recommandations seront personnalisées")
with st.form("user_info", clear_on_submit=True):
    st.write("Renseignez votre n° d'utilisateur")
    user_id_input = st.text_input("Numéro utilisateur")
    st.write("Indiquez le titre du film sur lequel faire nos recommandations")
    title = st.text_input("Film")
    submitted = st.form_submit_button("Soumettre", use_container_width= True )

    if submitted:
        # Envoyer la requête POST
        response = requests.post(
        "http://fastapi:6060/predict/identified_user",
        json={"userId": user_id_input, "movie_title": title}
        )

        # Vérifier la réponse
        if response.status_code == 200:
            result = response.json()
            st.write("Film sélectionné :")
            user_choice = result['user_choice']
            caption = f"{user_choice['title']}"
            st.image(user_choice["cover"], caption=caption, width=180)
            st.write("Nos 10 recommandations :")
            recommended_movies = result['recommendations']

            # Créer des colonnes pour afficher les 5 premières recommandations
            cols_recommended_movies = st.columns(5)
            for i, movie in enumerate(recommended_movies[:5]):
                col_index = i % 5
                with cols_recommended_movies[col_index]:
                    caption = f"{movie['title']}"
                    st.image(movie["cover"], caption=caption, use_column_width=True)

            # Ajouter une ligne horizontale
            st.markdown("---")

            # Créer des colonnes pour afficher les 5 suivantes recommandations
            cols_recommended_movies_second_half = st.columns(5)
            for i, movie in enumerate(recommended_movies[5:10]):
                col_index = i % 5
                with cols_recommended_movies_second_half[col_index]:
                    caption = f"{movie['title']}"
                    st.image(movie["cover"], caption=caption, use_column_width=True)
        else:
            st.error(f"Erreur lors de la requête : {response.status_code} - {response.text}")