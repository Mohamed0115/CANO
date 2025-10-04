# news.py
import streamlit as st
import streamlit as st



def show_news():
    st.title("📰 Latest Exoplanet News")
    st.write("Stay up-to-date with NASA's latest discoveries and relevant research.")

    # Add your existing default news here...
    default_news = [
        {
            "title": "NASA Discovers Earth-Sized Exoplanet in Habitable Zone",
            "url": "https://exoplanets.nasa.gov/news/",
            "summary": "NASA's TESS mission has identified an Earth-sized planet orbiting a nearby star in its habitable zone."
        },
        {
            "title": "James Webb Space Telescope Detects Atmosphere on Distant Exoplanet",
            "url": "https://exoplanets.nasa.gov/news/",
            "summary": "Webb observations reveal signs of water vapor and possible clouds on a gas giant exoplanet."
        },
        {
            "title": "Kepler Legacy Data Re-analyzed",
            "url": "https://exoplanets.nasa.gov/news/",
            "summary": "Scientists revisited Kepler’s archive, discovering new planetary candidates hidden in old data."
        },
        # Add the new research paper
        {
            "title": "Exoplanet detection using machine learning (MNRAS 2022)",
            "url": "https://academic.oup.com/mnras/article/513/4/5505/6472249",
            "summary": "In this paper, Malik et al. (2022) propose a method combining tsfresh feature extraction and LightGBM, achieving high AUC and recall in classifying planet signals."
        },
        
        {
            "title": "MNRAS: Supervised machine learning classification of TESS exoplanet candidates",
            "url": "https://academic.oup.com/mnras/article/513/4/5505/6472249",
            "summary": "A 2022 paper in Monthly Notices of the Royal Astronomical Society describing how machine learning can classify TESS exoplanet candidates. This is directly relevant to your project."
        }
    ]

    for article in default_news:
        st.markdown(f"### [{article['title']}]({article['url']})")
        st.write(article["summary"])
        st.markdown("---")
        # Back to Home Button
    if st.button("🏠 Back to Home"):
        st.session_state["mode"] = "home"
        st.rerun()
    

    st.caption("🚀 News feed includes both NASA updates and recent peer-reviewed research.")
