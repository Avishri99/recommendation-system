from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import extract_features
from src.model_training import train_trend_model
from src.recommend_trending import recommend_trending_movies

def main():
    # Step 1: Load and preprocess data
    ratings, movies = load_and_preprocess_data()

    # Step 2: Extract features
    features = extract_features(ratings)

    # Step 3: Train the model
    model = train_trend_model(features)

    # Step 4: Get recommendations
    recommended_movies = recommend_trending_movies(features, model, movies)

    # Remove duplicate values
    recommended_movies = recommended_movies.drop_duplicates()
    # Step 5: Save output
    recommended_movies.to_csv("output/trending_movies.csv", index=False)
    print("\nRecommended Trending Movies:")
    print(recommended_movies.head())

if __name__ == "__main__":
    main()
