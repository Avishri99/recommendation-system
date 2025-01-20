import pandas as pd
def extract_features(ratings):
    """
    Generate features such as avg_rating, num_ratings, weekly trends, and growth rate.
    :param ratings: Ratings DataFrame
    :return: Feature DataFrame
    """
    # Aggregate rating statistics
    rating_stats = ratings.groupby('movie_id').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    # Weekly trends
    ratings['week'] = ratings['timestamp'].dt.isocalendar().week
    weekly_trends = ratings.groupby(['movie_id', 'week']).agg(
        weekly_ratings=('rating', 'count')
    ).reset_index()

    # Growth rate (percentage change in weekly ratings)
    weekly_trends['growth_rate'] = weekly_trends.groupby('movie_id')['weekly_ratings'].pct_change()

    # Merge features
    features = pd.merge(rating_stats, weekly_trends, on='movie_id', how='left')
    features = features.fillna(0)

    return features
