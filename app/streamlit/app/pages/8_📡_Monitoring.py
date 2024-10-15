import streamlit as st
import streamlit.components.v1 as components

if not st.session_state['is_logged_in']:
    st.warning("You need to be logged in to access this page.")
    st.stop()

st.markdown("<h1 style='text-align: center;'>Testing & Monitoring</h1>", unsafe_allow_html=True)

grafana_url = "http://127.0.0.1:3000/d/ddz3bcm35rqiod/reco-movies-monitoring?orgId=1&from=1728991477901&to=1729013077901"

# Dimensions personnalisées
width = 1000  # Largeur en pixels
height = 800  # Hauteur en pixels

# Intégrer le tableau de bord Grafana avec des dimensions personnalisées
components.html(
    f'<iframe src="{grafana_url}" width="{width}" height="{height}" frameborder="0"></iframe>',
    height=height,
    width=width,
    scrolling=True  # Permet le défilement si nécessaire
)