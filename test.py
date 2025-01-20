from src.recommendation_engine import get_movie_recommendations
import pandas as pd

# Load movies and ratings data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Test with Collaborative Filtering (e.g., user_id = 1)
user_id = 1
recommended_movies_collab = get_movie_recommendations(user_id=user_id, ratings=ratings, movies=movies, method='collaborative', n_recommendations=5)
print("Collaborative Filtering Recommendations:", recommended_movies_collab)

# Test with Content-Based Filtering (e.g., movie_id = 1)
movie_id = 1
recommended_movies_content = get_movie_recommendations(user_id=None, movie_id=movie_id, ratings=ratings, movies=movies, method='content', n_recommendations=5)
print("Content-Based Filtering Recommendations:", recommended_movies_content)

# Test with SVD Matrix Factorization (e.g., user_id = 1)
recommended_movies_svd = get_movie_recommendations(user_id=user_id, ratings=ratings, movies=movies, method='svd', n_recommendations=5)
print("SVD Matrix Factorization Recommendations:", recommended_movies_svd)
