import streamlit as st
import requests

if not st.session_state['is_logged_in']:
    st.warning("You need to be logged in to access this page.")
    st.stop()

with st.form("user_info_no", clear_on_submit=True):
    st.write("Indiquez le titre du film sur lequel faire nos recommandations")
    title = st.text_input("Film")
    submitted_no = st.form_submit_button("Soumettre", use_container_width= True)

    if submitted_no:
        response = requests.post(
        "http://fastapi:8000/predict/new_user",
        json={"movie_title": title}
        )

        if response.status_code == 200:
            result = response.json()
            st.write("Film sélectionné :")
            user_choice = result['user_choice']
            caption = f"{user_choice['title']}"
            st.image(user_choice["cover"], caption=caption)
            st.write("Nos 10 recommandations :")
            recommended_movies = result['recommendations']

            cols_recommended_movies = st.columns(5)
            for i, movie in enumerate(recommended_movies):
                col_index = i % 5
                with cols_recommended_movies[col_index]:
                        caption = f"{movie['title']}"
                        st.image(movie["cover"], caption=caption, use_column_width=True)
        else:
            st.error(f"Erreur lors de la requête : {response.status_code} - {response.text}")