import streamlit as st
from researcher import show_researcher
from curious import show_curious
from news import show_news

# Landing page setup
st.set_page_config(page_title="Moon & Earth - Exoplanet Hunt", layout="wide")

if "mode" not in st.session_state:
    st.session_state["mode"] = "home"

# ---- Navigation (Landing page buttons) ----
if st.session_state["mode"] == "home":
    st.title("🌌 Moon & Earth: Exoplanet Hunt")
    st.write("Double click to Choose your journey & to go back choose \"Back to Home\" in the buttom")
    if st.button("🔬 Researcher"):
        st.session_state["mode"] = "researcher"
    if st.button("🌍 Curious Explorer"):
        st.session_state["mode"] = "curious"
    if st.button("📰 News"):
        st.session_state["mode"] = "news"

# ---- Researcher ----
elif st.session_state["mode"] == "researcher":
    show_researcher()

# ---- Curious ----
elif st.session_state["mode"] == "curious":
    show_curious()

# ---- News ----
elif st.session_state["mode"] == "news":
    show_news()
