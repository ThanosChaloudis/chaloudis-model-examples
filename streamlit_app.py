import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from fbprophet import Prophet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="ML Model Portfolio", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Churn Prediction", "Revenue Prediction", "DAU Forecast", "Recommender System", "Lootbox Simulation", "Customer Segmentation"])

# Main content
if page == "Welcome":
    st.title("Welcome to My Machine Learning Portfolio")
    st.write("This app showcases various machine learning models and techniques. Click on a model below to explore!")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Churn Prediction"):
            st.experimental_set_query_params(page="Churn Prediction")
        if st.button("Revenue Prediction"):
            st.experimental_set_query_params(page="Revenue Prediction")
    with col2:
        if st.button("DAU Forecast"):
            st.experimental_set_query_params(page="DAU Forecast")
        if st.button("Recommender System"):
            st.experimental_set_query_params(page="Recommender System")
    with col3:
        if st.button("Lootbox Simulation"):
            st.experimental_set_query_params(page="Lootbox Simulation")
        if st.button("Customer Segmentation"):
            st.experimental_set_query_params(page="Customer Segmentation")

elif page == "Churn Prediction":
    st.title("Churn Prediction Model")
    # Churn prediction code here

elif page == "Revenue Prediction":
    st.title("Revenue Prediction Model")
    # Revenue prediction code here

elif page == "DAU Forecast":
    st.title("Daily Active Users (DAU) Forecast")
    # DAU forecast code here

elif page == "Recommender System":
    st.title("Movie Recommender System")
    # Recommender system code here

elif page == "Lootbox Simulation":
    st.title("Lootbox Simulation")
    # Lootbox simulation code here

elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    # Customer segmentation code here

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