import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_movie_data

def get_recommendations(user_id, data, similarity_matrix, n_recommendations=5):
    user_ratings = data.loc[user_id]
    similar_users = similarity_matrix[data.index.get_loc(user_id)]
    
    # Get top similar users (excluding the user themselves)
    top_similar_users = data.index[np.argsort(similar_users)[::-1][1:11]]
    
    # Get movies the user hasn't rated
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Calculate predicted ratings
    predicted_ratings = {}
    for movie in unrated_movies:
        ratings = data.loc[top_similar_users, movie]
        similarities = similar_users[data.index.get_loc(top_similar_users)]
        predicted_ratings[movie] = np.average(ratings, weights=similarities)
    
    # Sort and return top recommendations
    sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    return sorted_ratings[:n_recommendations]

def show_recommender_system():
    st.title("Movie Recommender System")
    st.write("This is a simple movie recommender system using collaborative filtering.")

    data = generate_movie_data()

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(data)

    # User interface
    st.subheader("Movie Recommender")
    user_id = st.selectbox("Select a user", data.index.tolist())
    n_recommendations = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(user_id, data, similarity_matrix, n_recommendations)
        st.write(f"Top {n_recommendations} movie recommendations for {user_id}:")
        for movie, rating in recommendations:
            st.write(f"{movie}: Predicted rating {rating:.2f}")

    # Show user's current ratings
    st.subheader("User's Current Ratings")
    user_ratings = data.loc[user_id]
    non_zero_ratings = user_ratings[user_ratings > 0].sort_values(ascending=False)
    
    if len(non_zero_ratings) > 0:
        st.write(pd.DataFrame({'Movie': non_zero_ratings.index, 'Rating': non_zero_ratings.values}))
    else:
        st.write("This user hasn't rated any movies yet.")

    # Visualize user ratings distribution
    st.subheader("User Ratings Distribution")
    fig, ax = plt.subplots()
    user_ratings.hist(bins=5, ax=ax)
    ax.set_title(f"Rating Distribution for {user_id}")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Show global movie popularity
    st.subheader("Global Movie Popularity")
    movie_popularity = data.mean().sort_values(ascending=False)
    st.write(pd.DataFrame({'Movie': movie_popularity.index, 'Average Rating': movie_popularity.values}))