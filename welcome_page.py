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
        st.write("- Python (VSC)\n- R\n- Snowflake SQL")
    with col2:
        st.subheader("Visualization")
        st.write("- PowerBI\n- Tableau\n- Looker")
    with col3:
        st.subheader("Gaming Specific")
        st.write("- Game Design\n- Market Knowledge\n- Trends & Personal interest")


    st.header('Overview of Video Games I have applied my models in')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Wargaming')
        st.write('- World of Tanks\n- Unannounced Project')
    with col2:
        st.subheader('InnoGames')
        st.write('- Forge of Empires\n- Rise of Cultures\n- Elvenar')


    # Contact information
    st.header("Contact Me")
    st.write("""
    - Email: chaloudis.th@gmail.com
    - LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/achaloudis)
    """)

    # Optional: Add a call-to-action
    st.button("Download Resume", on_click=lambda: st.markdown("[Download Resume](https://drive.google.com/file/d/13ZOzc3LWLf4NK3UFhV4a3RTLLLtTO0ge/view)"))

if __name__ == "__main__":
    show_welcome_page()