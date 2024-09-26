import streamlit as st

# Initialisation de l'état de connexion si ce n'est pas déjà fait
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

st.markdown("<h1 style='text-align: center;'>Gestion MONGO Db</h1>", unsafe_allow_html=True)