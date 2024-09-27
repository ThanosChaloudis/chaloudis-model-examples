import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from utils import generate_dau_data

def show_dau_forecast():
    st.title("Daily Active Users (DAU) Forecast")
    st.write("This model forecasts Daily Active Users (DAU) using Facebook Prophet.")

    data = generate_dau_data()

    # Select historical data range
    st.subheader("Select historical data range")
    start_date = st.date_input("Start date", min(data['ds']).date())
    end_date = st.date_input("End date", max(data['ds']).date())

    filtered_data = data[(data['ds'].dt.date >= start_date) & (data['ds'].dt.date <= end_date)]

    # Train Prophet model
    model = Prophet()
    model.fit(filtered_data)

    # Make future predictions
    future_days = st.slider("Number of days to forecast", 30, 365, 90)
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)

    # Plot results
    st.subheader("DAU Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Show forecast data
    st.subheader("Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Additional analysis
    st.subheader("Additional Analysis")
    
    # Weekly pattern
    st.write("Weekly Pattern")
    weekly_pattern = forecast.groupby(forecast.ds.dt.dayofweek)['yhat'].mean().sort_index()
    fig, ax = plt.subplots()
    ax.bar(weekly_pattern.index, weekly_pattern.values)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Predicted DAU')
    ax.set_title('Predicted Weekly DAU Pattern')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    st.pyplot(fig)

    # Monthly trend
    st.write("Monthly Trend")
    monthly_trend = forecast.set_index('ds').resample('M')['yhat'].mean()
    fig, ax = plt.subplots()
    ax.plot(monthly_trend.index, monthly_trend.values)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Predicted DAU')
    ax.set_title('Predicted Monthly DAU Trend')
    plt.xticks(rotation=45)
    st.pyplot(fig)