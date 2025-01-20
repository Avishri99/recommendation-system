# recommendation_engine.py

from .collaborative_filtering import collaborative_filtering
from .content_based_filtering import content_based_filtering
from .svd_matrix_factorization import svd_matrix_factorization


def get_movie_recommendations(user_id=None, movie_id=None, ratings=None, movies=None, method='collaborative', n_recommendations=5):
    """
    Integrates all recommendation algorithms and returns the recommendations.
    
    :param user_id: ID of the user for collaborative filtering (optional)
    :param movie_id: ID of the movie for content-based filtering (optional)
    :param ratings: DataFrame containing user-item ratings
    :param movies: DataFrame containing movie metadata
    :param method: Recommendation method ('collaborative', 'content', or 'svd')
    :param n_recommendations: Number of recommendations to return
    :return: List of recommended movie IDs
    """
    
    if method == 'collaborative':
        return collaborative_filtering(user_id, ratings, n_recommendations)
    
    elif method == 'content':
        return content_based_filtering(movie_id, movies, ratings, n_recommendations)
    
    elif method == 'svd':
        return svd_matrix_factorization(user_id, ratings, n_recommendations)
    
    else:
        raise ValueError("Method must be 'collaborative', 'content', or 'svd'")
