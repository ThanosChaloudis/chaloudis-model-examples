import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utils import generate_churn_data

def show_churn_prediction():
    st.title("Churn Prediction Model")
    st.write("This model predicts customer churn using either XGBoost or Random Forest.")

    data = generate_churn_data()

    # Model selection
    model_type = st.selectbox("Select Model", ["XGBoost", "Random Forest"])

    # Train-test split
    X = data.drop('churn', axis=1)
    y = data['churn']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    # Model evaluation
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    st.write(f"Train Accuracy: {train_accuracy:.2f}")
    st.write(f"Test Accuracy: {test_accuracy:.2f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    st.write("Feature Importance:")
    st.bar_chart(feature_importance.set_index('feature'))

    # Prediction on new data
    st.subheader("Predict Churn for a New Customer")
    new_customer = {}
    for col in X.columns:
        if col.startswith(('usage_frequency', 'total_spend', 'customer_service_calls')):
            new_customer[col] = st.number_input(f"Enter {col}", value=X[col].mean())
        elif col.startswith('contract_length'):
            new_customer[col] = st.selectbox(f"Select {col}", [0, 1])

    new_customer_df = pd.DataFrame([new_customer])
    prediction = model.predict_proba(new_customer_df)[0]

    st.write(f"Probability of churn: {prediction[1]:.2f}")