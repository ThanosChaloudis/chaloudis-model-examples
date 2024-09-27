import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils import generate_revenue_data

def show_revenue_prediction():
    st.title("Forecasting Future Fortunes: Revenue Prediction")

    st.write("""
    Welcome to StreamFlix's finance department! As our streaming service grows, 
    we need to predict our future revenue. This helps us make important decisions 
    about hiring, content production, and expansion plans.

    Let's take a look at some of the data we use to make these predictions:
    """)

    data = generate_revenue_data()
    st.dataframe(data.head())

    st.write("""
    In this table:
    - 'advertising_spend' is how much we spent on ads
    - 'website_traffic' is the number of visitors to our site
    - 'customer_satisfaction' is our average rating out of 5
    - 'product_quality' is how we rate our content library
    - 'market_demand' represents overall interest in streaming services
    - 'competitor_prices' is the average price of our competitors
    - 'seasonal_factor' accounts for how the time of year affects us
    - 'revenue' is our target - what we're trying to predict!

    Now, let's use this data to predict our future revenue. We have two smart 
    calculators to choose from: Linear Regression and Random Forest.

    - Linear Regression is like drawing a straight line through our data points
    - Random Forest is like asking a group of experts and taking their average opinion

    You can choose which features you think are most important, and which calculator to use!
    """)

    # Feature selection
    st.subheader("What factors should we consider?")
    selected_features = st.multiselect("Choose factors", data.columns[:-1].tolist(), default=data.columns[:3].tolist())

    # Model selection
    model_type = st.selectbox("Which calculator should we use?", ["Linear Regression", "Random Forest"])

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

    st.write(f"""
    Great! Our {model_type} calculator has crunched the numbers. 
    It can explain about {test_r2:.2%} of the variation in our revenue based on these factors.
    That's pretty good, but remember, predicting the future is never 100% accurate!
    """)

    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        st.write("Here's how important each factor is in making our prediction:")
        st.bar_chart(feature_importance.set_index('feature'))

    # Visualize actual vs predicted
    st.subheader("How accurate are our predictions?")
    st.write("Let's compare our predictions to the actual revenue:")

    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Actual vs Predicted Revenue")
    st.pyplot(fig)

    st.write("""
    In this chart:
    - Each blue dot represents a prediction
    - The red line shows where perfect predictions would fall
    - Dots close to the line mean more accurate predictions
    """)

    # Prediction on new data
    st.subheader("Let's Predict Future Revenue!")
    st.write("Now, let's use our calculator to predict revenue based on new data:")

    new_data = {}
    for feature in selected_features:
        new_data[feature] = st.number_input(f"Enter {feature}", value=X[feature].mean())

    new_data_df = pd.DataFrame([new_data])
    prediction = model.predict(new_data_df)[0]

    st.write(f"""
    Based on this information, our {model_type} calculator predicts a revenue of ${prediction:,.2f}.

    Remember, this is just an estimate. Many factors can affect our actual revenue, 
    including some that might not be in our data. Always use predictions as a guide, 
    not a guarantee!
    """)