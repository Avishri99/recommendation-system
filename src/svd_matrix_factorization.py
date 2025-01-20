# svd_matrix_factorization.py

import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np

def svd_matrix_factorization(user_id, ratings, n_recommendations=5):
    """
    Recommends movies based on SVD (Matrix Factorization).
    
    :param user_id: ID of the user to recommend movies for
    :param ratings: DataFrame containing user-item ratings
    :param n_recommendations: Number of recommendations to return
    :return: List of movie IDs
    """
    
    # Create the user-item matrix
    user_movie_ratings = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
    
    # Fill NaN values with 0 (or use other strategies like mean)
    user_movie_ratings = user_movie_ratings.fillna(0)
    
    # Apply Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=10, random_state=42)
    matrix_factorization = svd.fit_transform(user_movie_ratings)
    
    # Compute the dot product of the user vector with all movie vectors
    user_vector = matrix_factorization[user_id - 1]  # Adjust index (user_id starts from 1)
    predicted_ratings = np.dot(user_vector, svd.components_)
    
    # Get the movie IDs with the highest predicted ratings
    movie_ids = user_movie_ratings.columns
    recommended_movies = [movie_id for movie_id, rating in zip(movie_ids, predicted_ratings) if rating > 0]
    
    return recommended_movies[:n_recommendations]
