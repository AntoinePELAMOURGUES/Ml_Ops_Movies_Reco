import streamlit as st

if not st.session_state['is_logged_in']:
    st.warning("You need to be logged in to access this page.")
    st.stop()

st.markdown("<h1 style='text-align: center;'>Testing & Monitoring</h1>", unsafe_allow_html=True)

grafana_url = "http://127.0.0.1:3000/d/ddz3bcm35rqiod/reco-movies-monitoring?orgId=1&from=1728991477901&to=1729013077901"

# Embed the Grafana dashboard using an iframe
st.markdown(
    f'<iframe src="{grafana_url}" width="900" height="700" frameborder="0"></iframe>',
    unsafe_allow_html=True
)