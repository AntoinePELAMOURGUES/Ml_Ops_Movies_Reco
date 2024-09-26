import streamlit as st

if not st.session_state['is_logged_in']:
    st.warning("You need to be logged in to access this page.")
    st.stop()

st.markdown("<h1 style='text-align: center;'>Testing & Monitoring</h1>", unsafe_allow_html=True)