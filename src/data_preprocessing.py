import pandas as pd
import numpy as np



def load_and_preprocess_data():
    ratings = pd.read_csv("data/ratings.csv")
    ratings = add_dummy_timestamp(ratings)
    print(ratings.head())
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies


def add_dummy_timestamp(ratings):
    # Add a dummy 'timestamp' column with random dates
    num_rows = len(ratings)
    ratings['timestamp'] = pd.to_datetime(
        np.random.choice(pd.date_range('2023-01-01', '2023-12-31'), num_rows)
    )
    return ratings


    # Convert timestamp to datetime
    # ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    return ratings, movies
