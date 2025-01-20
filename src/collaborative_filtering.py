# collaborative_filtering.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering(user_id, ratings, n_recommendations=5):
    """
    Recommends movies based on collaborative filtering (user-item).
    
    :param user_id: ID of the user to recommend movies for
    :param ratings: DataFrame containing user-item ratings
    :param n_recommendations: Number of recommendations to return
    :return: List of movie IDs
    """
    
    # Create the user-item matrix
    user_movie_ratings = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
    
    # Fill NaN values with 0 (or use other strategies like mean)
    user_movie_ratings = user_movie_ratings.fillna(0)
    
    # Compute cosine similarity between users
    similarity_matrix = cosine_similarity(user_movie_ratings)
    
    # Get the similarity scores for the given user
    user_idx = user_movie_ratings.index.get_loc(user_id)
    similarity_scores = similarity_matrix[user_idx]
    
    # Get the movies rated by the most similar users
    similar_users = similarity_scores.argsort()[-n_recommendations-1:-1][::-1]
    
    recommended_movies = []
    for similar_user in similar_users:
        similar_user_ratings = user_movie_ratings.iloc[similar_user]
        rated_movies = similar_user_ratings[similar_user_ratings > 0].index.tolist()
        recommended_movies.extend(rated_movies)
    
    # Remove movies already rated by the user
    user_rated_movies = user_movie_ratings.loc[user_id][user_movie_ratings.loc[user_id] > 0].index.tolist()
    recommended_movies = list(set(recommended_movies) - set(user_rated_movies))
    
    return recommended_movies[:n_recommendations]
