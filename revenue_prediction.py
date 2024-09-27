import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils import generate_revenue_data

def show_revenue_prediction():
    st.title("Revenue Prediction Model")
    st.write("This model predicts revenue based on selected features.")

    data = generate_revenue_data()

    # Feature selection
    st.subheader("Select features for the model")
    selected_features = st.multiselect("Choose features", data.columns[:-1].tolist(), default=data.columns[:3].tolist())

    # Model selection
    model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

    # Prepare data
    X = data[selected_features]
    y = data['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)

    # Model evaluation
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    st.write(f"Train R-squared: {train_r2:.2f}")
    st.write(f"Test R-squared: {test_r2:.2f}")

    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        st.write("Feature Importance:")
        st.bar_chart(feature_importance.set_index('feature'))

    # Prediction on new data
    st.subheader("Predict Revenue for New Data")
    new_data = {}
    for feature in selected_features:
        new_data[feature] = st.number_input(f"Enter {feature}", value=X[feature].mean())

    new_data_df = pd.DataFrame([new_data])
    prediction = model.predict(new_data_df)[0]

    st.write(f"Predicted Revenue: ${prediction:.2f}")