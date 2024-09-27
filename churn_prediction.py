import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utils import generate_churn_data

def show_churn_prediction():
    st.title("Churn Prediction: Keeping Our Customers Happy")

    st.write("""
    Let's imagine we run a subscription-based streaming service called "StreamFlix". 
    Our customer database looks something like this:
    """)

    data = generate_churn_data()
    st.dataframe(data.head())

    st.write("""
    In this table:
    - 'usage_frequency' shows how often a customer uses our service per month
    - 'contract_length' indicates if they're on a monthly or yearly plan
    - 'total_spend' is how much they've spent with us so far
    - 'customer_service_calls' is the number of times they've contacted our support
    - 'churn' is whether they've left our service (1) or stayed (0)

    As a business, we want to keep our customers happy and subscribed. The challenge is 
    to predict which customers might leave us (we call this 'churn') before they actually do. 
    If we can predict this, we can take steps to keep them with us!

    To solve this, we're going to use a clever computer program called XGBoost (or Random Forest, 
    if you prefer). Think of it as a super-smart detective that looks at all our customer data 
    and figures out patterns that lead to customers leaving.
    """)

    # Model selection
    model_type = st.selectbox("Which super-smart detective should we use?", ["XGBoost", "Random Forest"])

    st.write(f"Great! Let's see how our {model_type} detective does.")

    # Prepare the data
    X = data.drop('churn', axis=1)
    y = data['churn']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    st.write(f"""
    Our {model_type} detective has finished its investigation! 
    It correctly identified churning customers {test_accuracy:.2%} of the time on new data it hadn't seen before.
    That's pretty smart!
    """)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    st.write("""
    Now, let's see what our detective thinks are the most important clues for predicting if a customer will leave:
    """)
    st.bar_chart(feature_importance.set_index('feature'))

    st.write("""
    The taller the bar, the more important that clue is for predicting if a customer might leave us.
    """)

    # Prediction on new data
    st.subheader("Will This Customer Stay or Go?")
    st.write("Let's use our smart detective on a new customer and see what it predicts!")

    new_customer = {}
    for col in X.columns:
        if col.startswith(('usage_frequency', 'total_spend', 'customer_service_calls')):
            new_customer[col] = st.number_input(f"Enter {col}", value=X[col].mean())
        elif col.startswith('contract_length'):
            new_customer[col] = st.selectbox(f"Select {col}", [0, 1])

    new_customer_df = pd.DataFrame([new_customer])
    prediction = model.predict_proba(new_customer_df)[0]

    st.write(f"""
    Based on this information, our {model_type} detective thinks there's a {prediction[1]:.2%} chance 
    this customer might leave us. 

    If this number is high, we might want to send them a special offer or check in with them to make sure they're happy!
    """)