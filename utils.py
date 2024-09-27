import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def generate_churn_data():
    np.random.seed(42)
    n_samples = 1000
    features = pd.DataFrame({
        'usage_frequency': np.random.randint(1, 31, n_samples),
        'contract_length': np.random.choice(['monthly', 'yearly'], n_samples),
        'total_spend': np.random.uniform(10, 1000, n_samples),
        'customer_service_calls': np.random.randint(0, 10, n_samples)
    })
    churn = (features['usage_frequency'] < 10) | (features['total_spend'] < 100)
    features['churn'] = churn.astype(int)
    return features

@st.cache_data
def generate_revenue_data():
    np.random.seed(42)
    n_samples = 1000
    features = pd.DataFrame({
        'advertising_spend': np.random.uniform(1000, 10000, n_samples),
        'website_traffic': np.random.randint(1000, 100000, n_samples),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples),
        'product_quality': np.random.uniform(1, 5, n_samples),
        'market_demand': np.random.uniform(0, 1, n_samples),
        'competitor_prices': np.random.uniform(50, 200, n_samples),
        'seasonal_factor': np.random.uniform(0.8, 1.2, n_samples)
    })
    revenue = (
        2000 +
        5 * features['advertising_spend'] +
        0.1 * features['website_traffic'] +
        5000 * features['customer_satisfaction'] +
        7000 * features['product_quality'] +
        20000 * features['market_demand'] -
        50 * features['competitor_prices'] +
        10000 * features['seasonal_factor'] +
        np.random.normal(0, 5000, n_samples)
    )
    features['revenue'] = revenue
    return features

@st.cache_data
def generate_dau_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31')
    n_samples = len(dates)
    trend = np.linspace(1000, 5000, n_samples)
    seasonality = 500 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
    noise = np.random.normal(0, 100, n_samples)
    dau = trend + seasonality + noise
    return pd.DataFrame({'ds': dates, 'y': dau.astype(int)})

@st.cache_data
def generate_movie_data():
    np.random.seed(42)
    n_users = 100
    n_movies = 50
    ratings = np.random.randint(1, 6, size=(n_users, n_movies))
    user_ids = [f"User_{i}" for i in range(n_users)]
    movie_ids = [f"Movie_{i}" for i in range(n_movies)]
    return pd.DataFrame(ratings, index=user_ids, columns=movie_ids)

@st.cache_data
def generate_customer_data():
    np.random.seed(42)
    n_customers = 1000
    
    # Generate features
    recency = np.random.randint(1, 365, n_customers)  # days since last purchase
    frequency = np.random.randint(1, 50, n_customers)  # number of purchases
    monetary = np.random.uniform(10, 1000, n_customers)  # total spend
    
    # Create dataframe
    df = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })
    
    return df