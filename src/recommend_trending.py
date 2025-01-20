import pandas as pd
def recommend_trending_movies(features, model, movies):
    """
    Recommend trending movies using the trained model.
    :param features: DataFrame with features
    :param model: Trained trend prediction model
    :param movies: Movies DataFrame
    :return: Recommended trending movies DataFrame
    """
    # Predict trending movies
    features['predicted_trending'] = model.predict(features[['avg_rating', 'num_ratings', 'growth_rate']])

    # Filter trending movies
    trending_movies = features[features['predicted_trending'] == 1]

    # Merge with movies dataset
    recommended_movies = pd.merge(trending_movies, movies, on='movie_id')

    # Sort by average rating
    recommended_movies = recommended_movies.sort_values(by='avg_rating', ascending=False)

    return recommended_movies[['movie_id', 'movie_title', 'avg_rating']]
