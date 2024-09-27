import streamlit as st

def show_welcome_page():
    st.title("Machine Learning Portfolio")
    st.write("Welcome to my Machine Learning Portfolio! This application showcases various ML models and techniques.")

    # Introduction
    st.header("About Me")
    st.write("""
    Hello! I'm [Your Name], a data scientist and machine learning enthusiast. 
    This portfolio demonstrates my skills in developing and implementing various machine learning models.
    Feel free to explore the different projects and don't hesitate to reach out if you have any questions!
    """)

    # Skills section
    st.header("Skills")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Programming")
        st.write("- Python\n- R\n- SQL")
    with col2:
        st.subheader("ML / DL")
        st.write("- Scikit-learn\n- TensorFlow\n- PyTorch")
    with col3:
        st.subheader("Tools")
        st.write("- Git\n- Docker\n- AWS")

    # Projects overview
    st.header("Projects Overview")
    st.write("Click on a project to explore it in detail:")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("[Churn Prediction](/?page=Churn+Prediction)")
        st.write("Predict customer churn using classification models.")
        
        st.subheader("[Revenue Prediction](/?page=Revenue+Prediction)")
        st.write("Forecast revenue using regression techniques.")
        
        st.subheader("[DAU Forecast](/?page=DAU+Forecast)")
        st.write("Predict Daily Active Users using time series analysis.")
    
    with col2:
        st.subheader("[Recommender System](/?page=Recommender+System)")
        st.write("Movie recommendations using collaborative filtering.")
        
        st.subheader("[Lootbox Simulation](/?page=Lootbox+Simulation)")
        st.write("Simulate lootbox mechanics and analyze probabilities.")
        
        st.subheader("[Customer Segmentation](/?page=Customer+Segmentation)")
        st.write("Segment customers using clustering algorithms.")

    # Contact information
    st.header("Contact Me")
    st.write("""
    - Email: your.email@example.com
    - LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/yourprofile)
    - GitHub: [Your GitHub Profile](https://github.com/yourusername)
    """)

    # Optional: Add a call-to-action
    st.button("Download Resume", on_click=lambda: st.markdown("[Download Resume](link_to_your_resume.pdf)"))

if __name__ == "__main__":
    show_welcome_page()