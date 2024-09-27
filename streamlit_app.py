import streamlit as st
from welcome_page import show_welcome_page
from churn_prediction import show_churn_prediction
from revenue_prediction import show_revenue_prediction
# from dau_forecast import show_dau_forecast
from recommender_system import show_recommender_system
from lootbox_simulation import show_lootbox_simulation
from customer_segmentation import show_customer_segmentation

# Set page config
st.set_page_config(page_title="ML Model Portfolio", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Churn Prediction", "Revenue Prediction", "DAU Forecast", "Recommender System", "Lootbox Simulation", "Customer Segmentation"])

# Main content
if page == "Welcome":
    show_welcome_page()

elif page == "Churn Prediction":
    show_churn_prediction()

elif page == "Revenue Prediction":
    show_revenue_prediction()

# elif page == "DAU Forecast":
#     show_dau_forecast()

elif page == "Recommender System":
    show_recommender_system()

elif page == "Lootbox Simulation":
    show_lootbox_simulation()

elif page == "Customer Segmentation":
    show_customer_segmentation()

# Add any global styles or custom CSS here
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    height: 100px;
    font-size: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)