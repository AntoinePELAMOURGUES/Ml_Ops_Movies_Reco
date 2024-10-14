import streamlit as st
import requests


# Setup web page
st.set_page_config(
     page_title="API de Recommndation de films",
     page_icon="👋",
)

# Initialisation de l'état de connexion si ce n'est pas déjà fait
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False


st.markdown("<h1 style='text-align: center;'>PROJET ML_Ops</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>RECOMMANDATION DE FILMS</h2>", unsafe_allow_html=True)

st.markdown('---')
st.image("./images/datascientest.png", width=500)

# Création et mise en forme de notre Sidebar

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
st.sidebar.write(":red[MENTOR :]")
st.sidebar.write("Maria")
