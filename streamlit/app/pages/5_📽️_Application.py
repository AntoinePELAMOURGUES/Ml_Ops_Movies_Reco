import streamlit as st

if not st.session_state['is_logged_in']:
    st.warning("You need to be logged in to access this page.")
    st.stop()

st.markdown("<h1 style='text-align: center;'>Bienvenue sur notre site de recommandation de films</h1>", unsafe_allow_html=True)

logo = "/app/images/netflix-catalogue.jpg"

# Affichage de l'image en haut de la page
st.image(logo)

st.write("#### Possédez-vous un numéro d'utilisateur ?")

col1, col2 = st.columns(2, gap = 'large')

with col1:
    if st.button("Oui", use_container_width= True):
        st.switch_page("pages/7_🔭_ids_users.py")
with col2:
    if st.button("Non", use_container_width= True):
        st.switch_page("pages/6_🔍_new_user.py")

