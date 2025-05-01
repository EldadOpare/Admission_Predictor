import streamlit as st


home_page = st.Page(
    page = "app_pages/home_page.py",
    title = "Home Page",
    icon = "🏠"
)

upload_page = st.Page(
    page = "app_pages/upload.py",
    title = "Admission Pool Prediction",
    icon = "📚"
)



pg = st.navigation(
    {
        "Home" : [home_page],
        "Admission Predictor" : [upload_page],
    }
)

st.logo("Images/logo.jpeg",size = "large")

pg.run()